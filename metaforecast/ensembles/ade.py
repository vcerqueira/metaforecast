from typing import List, Optional, Tuple, Union

import lightgbm as lgb
import pandas as pd
from mlforecast import MLForecast
from sklearn.multioutput import MultiOutputRegressor as MIMO
from statsforecast import StatsForecast

from metaforecast.ensembles.base import BaseADE
from metaforecast.utils.normalization import Normalizations

DataFrameTuple = Tuple[pd.DataFrame, pd.DataFrame]
DataFrameLike = Union[pd.DataFrame, DataFrameTuple]


class ADE(BaseADE):
    """Arbitrated Dynamic Ensemble

    Dynamic ensemble approach where ensemble members are weighted based on
    a meta-model that forecasts their error. The ADE approach dynamically combines
    forecasts from multiple models by learning
    their error patterns. It's particularly effective when dealing with heterogeneous
    time series and when the relative performance of individual models varies over time.

    Parameters
    ----------
    freq : str
        String denoting the sampling frequency of the time series (e.g. "MS").
    trim_ratio : float, optional
        Ratio (0-1) of ensemble members to keep in the ensemble.
        (1-trim_ratio) of models will not be used during inference based on validation accuracy.
        Defaults to 1, which means all ensemble members are used.
    meta_lags : list of int
        List of lags to be used in the training of the meta-model.
        Follows the structure of mlforecast. Example: [1,2,3,8].
    trim_by_uid : bool, optional
        Whether to trim the ensemble by unique_id (True) or dataset (False).
        Defaults to True, but this can become computationally demanding for datasets with
        a large number of time series.
    meta_model : object, optional
        Learning algorithm to use in the meta-level to forecast the
        error of ensemble members. Defaults to a linear LGBM with a default configuration.

    References
    ----------
    Cerqueira, V., Torgo, L., Pinto, F., & Soares, C. (2019).
    Arbitrage of forecasting experts. Machine Learning, 108, 913-944.

    Examples
    --------
        >>> from datasetsforecast.m3 import M3
        >>> from neuralforecast import NeuralForecast
        >>> from neuralforecast.models import NHITS, NBEATS, MLP
        >>>
        >>> from metaforecast.ensembles import ADE
        >>>
        >>> df, *_ = M3.load('.', group='Monthly')
        >>>
        >>> # ensemble members setup
        >>> CONFIG = {'input_size': 12,
        >>>           'h': 12,
        >>>           'accelerator': 'cpu',
        >>>           'max_steps': 10, }
        >>>
        >>> models = [
        >>>     NBEATS(**CONFIG, stack_types=3 * ["identity"]),
        >>>     NHITS(**CONFIG),
        >>>     MLP(**CONFIG),
        >>>     MLP(num_layers=3, **CONFIG),
        >>> ]
        >>>
        >>> nf = NeuralForecast(models=models, freq='M')
        >>>
        >>> # cv to build meta-data
        >>> n_windows = df['unique_id'].value_counts().min()
        >>> n_windows = int(n_windows // 2)
        >>> fcst_cv = nf.cross_validation(df=df, n_windows=n_windows, step_size=1)
        >>> fcst_cv = fcst_cv.reset_index()
        >>> fcst_cv = fcst_cv.groupby(['unique_id', 'cutoff']).head(1).drop(columns='cutoff')
        >>>
        >>> # fitting combination rule
        >>> ensemble = ADE(freq='ME', meta_lags=list(range(1,7)), trim_ratio=0.6)
        >>> ensemble.fit(fcst_cv)
        >>>
        >>> # re-fitting models
        >>> nf.fit(df=df)
        >>>
        >>> # forecasting and combining
        >>> fcst = nf.predict()
        >>> fcst_ensemble = ensemble.predict(fcst.reset_index(), train=df, h=12)
    """

    _LGB_PARS = {"verbosity": -1, "n_jobs": 1, "linear_tree": True}
    _MLF_PREPROCESS_PARS = {"static_features": []}

    def __init__(
        self,
        freq: str,
        meta_lags: Optional[List[int]] = None,
        trim_ratio: float = 1,
        trim_by_uid: bool = True,
        meta_model=MIMO(lgb.LGBMRegressor(**_LGB_PARS)),
    ):
        self.frequency = freq

        super().__init__(
            window_size=self.WINDOW_SIZE_BY_FREQ[self.frequency],
            trim_ratio=trim_ratio,
            trim_by_uid=trim_by_uid,
            meta_model=meta_model,
        )

        self.model_names = None

        if meta_lags is None:
            n_lags = self.WINDOW_SIZE_BY_FREQ[self.frequency]
            self.meta_lags = list((1, n_lags + 1))
        else:
            self.meta_lags = meta_lags

        self.lag_names = [f"lag{i}" for i in self.meta_lags]

        self.meta_mlf = MLForecast(models=[], freq=self.frequency, lags=self.meta_lags)

        self.meta_df = None
        self.raw_meta_data = None
        self.insample_scores = None
        self.use_window = False
        self.weights = None

    # pylint: disable=arguments-differ
    def fit(self, insample_fcst: pd.DataFrame, **kwargs):
        """fit

        Parameters
        ----------
        insample_fcst : pd.DataFrame
            Forecast and actual values dataset formatted like mlforecast cross-validation output.
            Contains either:
            - In-sample forecasts (predictions on training data)
            - Cross-validation results (out-of-sample predictions)
            Expected columns:
            - unique_id (or other id_col): Identifier for each series
            - ds (or other time_col): Timestamp
            - y (or other target_col): Actual values
            - *model_name*: Predictions by a model with name *model_name*

        Returns
        -------
        self
            self, with a fitted self.meta_model

        """
        self._fit(insample_fcst)

    def _fit(self, insample_fcst):
        if self.model_names is None:
            self.model_names = insample_fcst.columns.to_list()
            self.model_names = [
                x for x in self.model_names if x not in self.METADATA + ["h"]
            ]

        self._set_n_models()

        in_sample_loss_df = self._get_insample_loss(insample_fcst)

        self.insample_scores = self.evaluate_base_fcst(
            insample_fcst=insample_fcst, use_window=self.use_window
        )

        self.raw_meta_data = self.meta_mlf.preprocess(
            in_sample_loss_df, **self._MLF_PREPROCESS_PARS
        )

        self.meta_df = self._process_meta_data(self.raw_meta_data)

        x, y = self.meta_df
        if y.isna().any().any():
            y = y.ffill().bfill()

        self.meta_model.fit(x, y)

    # pylint: disable=arguments-differ
    def predict(self, fcst: pd.DataFrame, train: pd.DataFrame, h: int, **kwargs):
        """Combine ensemble member forecasts using the meta-model.

        Parameters
        ----------
        fcst : pd.DataFrame
            Forecasts from individual ensemble members.
            Expected columns: ['unique_id', 'ds', 'model_name1', 'model_name2', ...]
        train : pd.DataFrame
            Training dataset used to compute recent lags for meta-model input.
            Expected columns: ['unique_id', 'ds', 'y']
        h : int
            Forecast horizon (number of future periods to predict)

        Returns
        -------
        pd.Series
            Combined ensemble forecasts for h periods ahead.

        """
        self._assert_fcst(fcst)

        ade_fcst = self._predict(preds=fcst, train=train, h=h)
        ade_fcst.name = self.alias

        return ade_fcst

    # pylint: disable=arguments-differ
    def update_weights(self, fcst: pd.DataFrame, **kwargs):
        """Update performance statistics of ensemble members based on recent forecasts.

        Used to identify and retain top-performing models for ensemble trimming.
        Updates internal statistics tracking each model's forecast accuracy.

        Parameters
        ----------
        fcst: pd.DataFrame
            Recent forecasts and actual values for ensemble members.
            Expected columns:
            - unique_id: Series identifier
            - ds: Timestamp
            - model_name: Predictions with model with name "model_name"
            - y: Actual values

        Notes
        -----
        Not implemented yet
        """
        raise NotImplementedError

    def _predict(self, preds: pd.DataFrame, train: pd.DataFrame, h: int):
        df_ext = train.merge(preds, on=["unique_id", "ds"], how="outer")
        df_ext = df_ext[self.METADATA]
        df_ext["y"] = df_ext["y"].fillna(value=-1)

        meta_dataset = self.meta_mlf.preprocess(df_ext, **self._MLF_PREPROCESS_PARS)

        self.weights = self._weights_by_uid(meta_dataset, h=h)

        fcst = preds.apply(lambda x: self._weighted_average(x, self.weights), axis=1)

        return fcst

    def _get_insample_loss(self, insample_fcst: pd.DataFrame):
        """_get_insample_loss

        Compute in-sample (CV) point-wise loss

        :param insample_fcst: validation predictions for each model and actual values (y)
        :type insample_fcst: pd.DataFrame

        :return: pd.DataFrame with point-wise error scores of each ensemble member
        across the validation set
        """
        in_sample_loss = []
        in_sample_uid = insample_fcst.copy().groupby("unique_id")
        for _, uid_df in in_sample_uid:
            for mod in self.model_names:
                uid_df[mod] = uid_df[mod] - uid_df["y"]

            in_sample_loss.append(uid_df)

        in_sample_loss_df = pd.concat(in_sample_loss)

        # first h forward
        # could average all horizons
        if "h" in in_sample_loss_df.columns:
            in_sample_loss_df = in_sample_loss_df.query("h==1").drop(columns=["h"])

        return in_sample_loss_df

    # pylint: disable=invalid-name
    def _process_meta_data(
        self, meta_data: pd.DataFrame, return_X_y: bool = True
    ) -> DataFrameLike:
        lag_locs = meta_data.columns.str.startswith("lag")
        lag_cols = meta_data.columns[lag_locs].to_list()

        if return_X_y:
            # pylint: disable=invalid-name
            X_meta, Y_meta = meta_data[lag_cols], meta_data[self.model_names]
            return X_meta, Y_meta

        meta_df = meta_data[lag_cols + self.model_names]

        return meta_df

    # pylint: disable=arguments-differ
    def _weights_by_uid(self, df: pd.DataFrame, h: int, **kwargs):
        top_overall = self._get_top_k(self.insample_scores.mean())
        top_by_uid = self.insample_scores.apply(self._get_top_k, axis=1)

        uid_weights = {}
        for uid, meta_uid_df in df.groupby("unique_id"):
            if h > 1:
                lags = meta_uid_df.head(-(h - 1)).tail(1)[self.lag_names]
            else:
                lags = meta_uid_df.tail(1)[self.lag_names]

            meta_pred = self.meta_model.predict(lags)
            meta_pred = pd.DataFrame(meta_pred, columns=self.model_names)

            weights = self._weights_from_errors(meta_pred)

            if self.trim_by_uid:
                poor_models = [x not in top_by_uid[uid] for x in weights.index]
            else:
                poor_models = [x not in top_overall for x in weights.index]

            weights[poor_models] = 0
            weights /= weights.sum()

            uid_weights[uid] = weights

        weights_df = pd.DataFrame(uid_weights).T
        weights_df.index.name = "unique_id"

        return weights_df

    def _reweight_by_redundancy(self):
        raise NotImplementedError

    # pylint: disable=arguments-renamed
    @staticmethod
    def _weights_from_errors(meta_predictions: pd.DataFrame) -> pd.Series:
        e_hat = meta_predictions.abs()

        weights = e_hat.apply(
            func=lambda x: Normalizations.normalize_and_proportion(-x), axis=1
        )

        weight_s = weights.iloc[0]

        return weight_s


class MLForecastADE(ADE):
    """Arbitrated Dynamic Ensemble (ADE) implemented for MLForecast.

    Creates a dynamic ensemble where member weights are determined by a meta-model
    that predicts each model's forecast error. The meta-model adapts weights
    based on recent performance and input patterns.

    Implementation follows the MLForecast conventions and models.
    To use ADE with statsforecast or neuralforecast, see :class:`ADE`.

    References
    ----------
    Cerqueira, V., Torgo, L., Pinto, F., & Soares, C. (2019).
    "Arbitrage of forecasting experts." Machine Learning, 108, 913-944.

    Examples
    --------
    >>> from datasetsforecast.m3 import M3
    >>> from mlforecast import MLForecast
    >>> from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> from sklearn.neighbors import KNeighborsRegressor
    >>>
    >>> from metaforecast.ensembles import MLForecastADE
    >>>
    >>> df, *_ = M3.load('.', group='Monthly')
    >>>
    >>> # ensemble members setup
    >>> models_ml = {
    >>>     'Ridge': RidgeCV(),
    >>>     'Lasso': LassoCV(),
    >>>     'Elastic-net': ElasticNetCV(),
    >>>     'DT': DecisionTreeRegressor(max_depth=5),
    >>>     'DStump': DecisionTreeRegressor(max_depth=1),
    >>>     'KNN': KNeighborsRegressor(n_neighbors=30),
    >>> }
    >>>
    >>> mlf = MLForecast(models=models_ml, freq='ME', lags=range(1, 7))
    >>>
    >>> # make sure fitted=True so we have samples for meta-learning
    >>> mlf.fit(df=df, fitted=True)
    >>>
    >>> # fitting combination rule
    >>> ensemble = MLForecastADE(mlf=mlf, trim_ratio=0.5)
    >>> ensemble.fit()
    >>>
    >>> fcst = ensemble.predict(train=df, h=12)

    """

    def __init__(
        self,
        mlf: MLForecast,
        sf: Optional[StatsForecast] = None,
        trim_ratio: float = 1,
        meta_model=MIMO(lgb.LGBMRegressor(**ADE._LGB_PARS)),
    ):
        """Initialize the Arbitrated Dynamic Ensemble with MLForecast models.

        Parameters
        ----------
        mlf : MLForecast
            Fitted MLForecast object containing ensemble members.
            Must be initialized with fitted=True to generate the meta-dataset
            for error prediction.

        sf : StatsForecast, optional
            StatsForecast object containing classical forecasting models to be
            included in the ensemble alongside MLForecast models.

        trim_ratio : float, default=1.0
            Proportion of ensemble members to retain, between 0 and 1.
            Models are selected based on validation accuracy:
            - 1.0 keeps all models
            - 0.5 keeps top 50% of models
            - Lower values create a more selective ensemble

        meta_model : object, optional
            Model used to predict ensemble members' errors.
            If None, defaults to LightGBM with linear trees and default parameters.

        Notes
        -----
        The meta_model should support scikit-learn's fit/predict API.

        """

        self.mlf = mlf
        self.sf = sf
        self.frequency = self.mlf.ts.freq

        super().__init__(
            freq=self.frequency,
            trim_ratio=trim_ratio,
            meta_model=meta_model,
            meta_lags=self.mlf.ts.lags,
        )

    # pylint: disable=arguments-differ
    def fit(self, **kwargs):
        """Train the meta-model using ensemble members' in-sample predictions.

        Uses cross-validation predictions from the MLForecast object to:
        1. Compute forecast errors for each ensemble member
        2. Train the meta-model to predict these errors
        3. Update model weights based on validation performance
        4. If trim_ratio < 1, selects top performing models

        """

        insample_fcst = self.mlf.fcst_fitted_values_

        if self.sf is not None:
            self.sf.forecast(fitted=True, h=1)
            insample_fcst_sf = self.sf.forecast_fitted_values()

            insample_fcst = insample_fcst.merge(
                insample_fcst_sf.drop(columns="y"), on=self.METADATA_NO_T
            )

        self._fit(insample_fcst)

    # pylint: disable=arguments-differ
    def predict(self, train: pd.DataFrame, h: int, **kwargs):
        """Generate ensemble forecasts using weighted model combinations.

        Parameters
        ----------
        train : pd.DataFrame
            Training dataset used to compute recent lags for meta-model input.
            Expected columns:
            - unique_id: Series identifier
            - ds: Timestamp
            - y: Target variable

        h : int
            Forecast horizon (number of periods to predict ahead)

        Returns
        -------
        pd.Series
            Combined ensemble predictions.

        """
        base_fcst = self.mlf.predict(h=h)

        if self.sf is not None:
            base_fcst_sf = self.sf.predict(h=h)

            base_fcst = base_fcst.merge(base_fcst_sf, on=self.METADATA_NO_T)

        fcst = self._predict(preds=base_fcst, train=train, h=h)

        return fcst

    def update_weights(self, fcst: pd.DataFrame, **kwargs):
        """Update performance statistics of ensemble members.

        See :meth:`ADE.update_weights` for full documentation.
        """
        raise NotImplementedError

    def _reweight_by_redundancy(self):
        raise NotImplementedError
