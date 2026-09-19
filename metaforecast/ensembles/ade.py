from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from mlforecast import MLForecast
from statsforecast import StatsForecast

from metaforecast.ensembles.base import BaseADE

DataFrameTuple = Tuple[pd.DataFrame, pd.DataFrame]
DataFrameLike = pd.DataFrame | DataFrameTuple


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
    h : int
        Forecast horizon (number of future periods to predict).
    meta_lags : int, list of int, or None
        Lags to be used in the training of the meta-model.
        If an integer ``n``, lags are set to ``list(range(1, n + 1))``.
        If a list, follows the structure of mlforecast. Example: [1,2,3,8].
        If None, ``n`` is taken from the frequency-based window size.
    trim_ratio : float, optional
        Ratio (0-1) of ensemble members to keep in the ensemble.
        (1-trim_ratio) of models will not be used during inference based on validation accuracy.
        Defaults to 1, which means all ensemble members are used.
    trim_by_uid : bool, optional
        Whether to trim the ensemble by unique_id (True) or dataset (False).
        Defaults to True, but this can become computationally demanding for datasets with
        a large number of time series.
    meta_model : object, optional
        Learning algorithm to use in the meta-level to forecast the
        error of ensemble members. Defaults to a CatBoost regressor.

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
        >>> ensemble = ADE(freq='ME', h=12, meta_lags=list(range(1,7)), trim_ratio=0.6)
        >>> ensemble.fit(fcst_cv)
        >>>
        >>> # re-fitting models
        >>> nf.fit(df=df)
        >>>
        >>> # forecasting and combining
        >>> fcst = nf.predict()
        >>> fcst_ensemble = ensemble.predict(fcst.reset_index(), train=df)
    """

    _CB_PARS = {
        "eval_metric": "MultiRMSE",
        "loss_function": "MultiRMSE",
        "od_type": "Iter",
        "allow_writing_files": False,
        "task_type": "CPU",
        "verbose": False,
    }

    _MLF_PREPROCESS_PARS = {"static_features": []}

    def __init__(
        self,
        freq: str,
        h: int,
        meta_lags: int | List[int] | None = None,
        trim_ratio: float = 1,
        trim_by_uid: bool = True,
        meta_model=None,
    ):
        self.frequency = freq
        self.h = h

        if meta_model is None:
            meta_model = CatBoostRegressor(**self._CB_PARS)

        super().__init__(
            window_size=self.WINDOW_SIZE_BY_FREQ[self.frequency],
            trim_ratio=trim_ratio,
            trim_by_uid=trim_by_uid,
            meta_model=meta_model,
        )

        self.model_names = None

        if self._is_lag_count(meta_lags):
            n_lags = (
                int(meta_lags)
                if meta_lags is not None
                else self.WINDOW_SIZE_BY_FREQ[self.frequency]
            )
            self.meta_lags = list(range(1, n_lags + 1))
        else:
            self.meta_lags = list(meta_lags)

        self.lag_names = [f"lag{i}" for i in self.meta_lags]

        self.meta_mlf = MLForecast(models=[], freq=self.frequency, lags=self.meta_lags)

        self.meta_df = None
        self.raw_meta_data = None
        self.insample_scores = None
        self.use_window = False
        self.weights = None

    @staticmethod
    def _is_lag_count(meta_lags) -> bool:
        if meta_lags is None:
            return True
        return isinstance(meta_lags, (int, np.integer)) and not isinstance(meta_lags, bool)

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
        return self

    def _fit(self, insample_fcst):
        if self.model_names is None:
            self.model_names = [c for c in insample_fcst.columns if c not in self.NON_MODEL_COLS]

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

    def predict(self, fcst: pd.DataFrame, train: pd.DataFrame, **kwargs):
        """Combine ensemble member forecasts using the meta-model.

        Parameters
        ----------
        fcst : pd.DataFrame
            Forecasts from individual ensemble members.
            Expected columns: ['unique_id', 'ds', 'model_name1', 'model_name2', ...]
        train : pd.DataFrame
            Training dataset used to compute recent lags for meta-model input.
            Expected columns: ['unique_id', 'ds', 'y']

        Returns
        -------
        pd.Series
            Combined ensemble forecasts for ``self.h`` periods ahead.

        """
        self._assert_fcst(fcst)

        ade_fcst = self._predict(preds=fcst, train=train)
        ade_fcst.name = self.alias

        return ade_fcst

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

    def _predict(self, preds: pd.DataFrame, train: pd.DataFrame):
        # Append forecast timestamps so lag1 at the origin is the last actual y.
        # MLForecast drops rows with null y even when dropna=False, so future y
        # is filled with a dummy; we then keep only the first forecast row per
        # series, whose lags still come from actuals.
        timeline = (
            train[self.METADATA]
            .merge(preds[["unique_id", "ds"]], on=["unique_id", "ds"], how="outer")
            .sort_values(["unique_id", "ds"])
        )
        timeline["y"] = timeline["y"].fillna(-1)
        meta_dataset = self.meta_mlf.preprocess(timeline, **self._MLF_PREPROCESS_PARS)

        origin = (
            preds[["unique_id", "ds"]]
            .sort_values(["unique_id", "ds"])
            .groupby("unique_id", sort=False)
            .head(1)
        )
        meta_at_origin = meta_dataset.merge(origin, on=["unique_id", "ds"])

        self.weights = self._weights_by_uid(meta_at_origin)

        return self._combine_forecasts(preds, self.weights)

    def _get_insample_loss(self, insample_fcst: pd.DataFrame):
        """_get_insample_loss

        Compute in-sample (CV) point-wise loss

        :param insample_fcst: validation predictions for each model and actual values (y)
        :type insample_fcst: pd.DataFrame

        :return: pd.DataFrame with point-wise error scores of each ensemble member
        across the validation set
        """
        loss_df = insample_fcst.copy()
        loss_df[self.model_names] = loss_df[self.model_names].sub(loss_df["y"], axis=0)

        # first h forward
        # could average all horizons
        if "h" in loss_df.columns:
            loss_df = loss_df.query("h==1").drop(columns=["h"])

        return loss_df

    def _process_meta_data(self, meta_data: pd.DataFrame, return_X_y: bool = True) -> DataFrameLike:
        lag_locs = meta_data.columns.str.startswith("lag")
        lag_cols = meta_data.columns[lag_locs].to_list()

        if return_X_y:
            X_meta, Y_meta = meta_data[lag_cols], meta_data[self.model_names]
            return X_meta, Y_meta

        meta_df = meta_data[lag_cols + self.model_names]

        return meta_df

    def _weights_by_uid(self, df: pd.DataFrame, **kwargs):
        latest = df.sort_values(["unique_id", "ds"]).groupby("unique_id", sort=False).tail(1)
        lags = latest[self.lag_names]
        meta_pred = pd.DataFrame(
            self.meta_model.predict(lags),
            columns=self.model_names,
            index=pd.Index(latest["unique_id"].to_numpy(), name="unique_id"),
        )

        weights = self._weights_from_errors(meta_pred)
        return self._apply_trim(weights, by_uid=self.trim_by_uid, scores=self.insample_scores)

    def _reweight_by_redundancy(self):
        raise NotImplementedError

    @staticmethod
    def _weights_from_errors(meta_predictions: pd.DataFrame) -> pd.DataFrame:
        neg = -meta_predictions.abs()
        row_min = neg.min(axis=1)
        span = neg.max(axis=1) - row_min
        scaled = neg.sub(row_min, axis=0).div(span.replace(0, np.nan), axis=0)
        n_models = scaled.shape[1]
        scaled = scaled.fillna(1.0 / n_models)
        return scaled.div(scaled.sum(axis=1), axis=0)


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
    >>> ensemble = MLForecastADE(mlf=mlf, h=12, trim_ratio=0.5)
    >>> ensemble.fit()
    >>>
    >>> fcst = ensemble.predict(train=df)

    """

    def __init__(
        self,
        mlf: MLForecast,
        h: int,
        sf: StatsForecast | None = None,
        trim_ratio: float = 1,
        meta_model=None,
    ):
        """Initialize the Arbitrated Dynamic Ensemble with MLForecast models.

        Parameters
        ----------
        mlf : MLForecast
            Fitted MLForecast object containing ensemble members.
            Must be initialized with fitted=True to generate the meta-dataset
            for error prediction.

        h : int
            Forecast horizon (number of future periods to predict).

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
            If None, defaults to a CatBoost regressor.

        Notes
        -----
        The meta_model should support scikit-learn's fit/predict API.

        """

        self.mlf = mlf
        self.sf = sf
        self.frequency = self.mlf.ts.freq

        super().__init__(
            freq=self.frequency,
            h=h,
            trim_ratio=trim_ratio,
            meta_model=meta_model,
            meta_lags=self.mlf.ts.lags,
        )

    def fit(self, **kwargs):
        """Train the meta-model using ensemble members' in-sample predictions.

        Uses cross-validation predictions from the MLForecast object to:
        1. Compute forecast errors for each ensemble member
        2. Train the meta-model to predict these errors
        3. Update model weights based on validation performance
        4. If trim_ratio < 1, selects top performing models

        Returns
        -------
        self
            self, with a fitted self.meta_model

        """

        insample_fcst = self.mlf.fcst_fitted_values_

        if self.sf is not None:
            self.sf.forecast(fitted=True, h=1)
            insample_fcst_sf = self.sf.forecast_fitted_values()

            insample_fcst = insample_fcst.merge(
                insample_fcst_sf.drop(columns="y"), on=self.METADATA_NO_T
            )

        self._fit(insample_fcst)
        return self

    def predict(self, train: pd.DataFrame, **kwargs):
        """Generate ensemble forecasts using weighted model combinations.

        Parameters
        ----------
        train : pd.DataFrame
            Training dataset used to compute recent lags for meta-model input.
            Expected columns:
            - unique_id: Series identifier
            - ds: Timestamp
            - y: Target variable

        Returns
        -------
        pd.Series
            Combined ensemble predictions.

        """
        base_fcst = self.mlf.predict(h=self.h)

        if self.sf is not None:
            base_fcst_sf = self.sf.predict(h=self.h)

            base_fcst = base_fcst.merge(base_fcst_sf, on=self.METADATA_NO_T)

        fcst = self._predict(preds=base_fcst, train=train)
        fcst.name = self.alias

        return fcst
