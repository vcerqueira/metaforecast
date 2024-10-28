from typing import Optional

import pandas as pd

from metaforecast.ensembles.base import ForecastingEnsemble


class Windowing(ForecastingEnsemble):
    """Combine forecasts based on recent performance in sliding windows.

    A dynamic ensemble that adapts model weights based on accuracy in
    recent time windows. This approach is particularly suited for
    non-stationary data where relative model performance changes over
    time [1], [2].

    References
    ----------
    [1] Cerqueira, V., Torgo, L., Oliveira, M., & Pfahringer, B. (2017).
    "Dynamic and heterogeneous ensembles for time series forecasting."
    In IEEE International Conference on Data Science and Advanced
    Analytics (DSAA) (pp. 242-251).

    [2] van Rijn, J. N., Holmes, G., Pfahringer, B., & Vanschoren, J. (2015).
    "Having a blast: Meta-learning and heterogeneous ensembles for data streams."
    In IEEE International Conference on Data Mining (pp. 1003-1008).

    Examples
    ---------
    >>> from datasetsforecast.m3 import M3
    >>> from neuralforecast import NeuralForecast
    >>> from neuralforecast.models import NHITS, NBEATS, MLP
    >>> from metaforecast.ensembles import Windowing
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
    >>> ensemble = Windowing(freq='ME', trim_ratio=.8)
    >>> ensemble.fit(fcst_cv)
    >>>
    >>> # re-fitting models
    >>> nf.fit(df=df)
    >>>
    >>> # forecasting and combining
    >>> fcst = nf.predict()
    >>> fcst_ensemble = ensemble.predict(fcst.reset_index())

    """

    def __init__(
        self,
        freq: str,
        select_best: bool = False,
        trim_ratio: float = 1,
        weight_by_uid: bool = False,
        window_size: Optional[int] = None,
    ):
        """Initialize window-based dynamic ensemble.

        Parameters
        ----------
        freq : str
            Time series sampling frequency (e.g., 'M' for monthly, 'D' for daily).
            Used to set default window size if not specified.

        select_best : bool, default=False
            Model selection strategy:
            - True: Use single best model from window
            - False: Use weighted combination of models
            Single best can be more aggressive in adapting to changes

        trim_ratio : float, default=1.0
            Proportion of models to retain in ensemble, between 0 and 1:
            - 1.0: Keep all models
            - 0.5: Keep top 50% of models
            Models are selected based on window performance

        weight_by_uid : bool, default=True
            Whether to compute weights separately for each series:
            - True: Individual weights per series (may be computationally intensive)
            - False: Global weights across all series

        window_size : int, optional
            Number of recent observations used for performance evaluation.
            If None, defaults to frequency-based size:
            - Monthly: 12 observations
            - Daily: 30 observations
            etc.

        """

        super().__init__()

        self.alias = "Windowing"
        self.frequency = freq

        if window_size is None:
            self.window_size = self.WINDOW_SIZE_BY_FREQ[self.frequency]
        else:
            self.window_size = window_size

        self.select_best = select_best
        if self.select_best:
            self.trim_ratio = 1e-10
            self.alias = "BLAST"
        else:
            self.trim_ratio = trim_ratio

        self.weight_by_uid = weight_by_uid
        self.insample_scores = None
        self.use_window = True

        self.weights = None

    # pylint: disable=arguments-differ
    def fit(self, insample_fcst, **kwargs):
        """Update performance statistics of ensemble members based on recent forecasts.

        Used to identify and retain top-performing models for ensemble trimming.
        Updates internal statistics tracking each model's forecast accuracy.

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
            self, with computed self.weights

        """
        if self.model_names is None:
            self.model_names = insample_fcst.columns.to_list()
            self.model_names = [
                x for x in self.model_names if x not in self.METADATA + ["h"]
            ]

        self._set_n_models()

        self.insample_scores = self.evaluate_base_fcst(
            insample_fcst=insample_fcst, use_window=self.use_window
        )

        self.weights = self._weights_by_uid()

    # pylint: disable=arguments-differ
    def predict(self, fcst: pd.DataFrame, **kwargs):
        """Combine ensemble member forecasts based on recent performance.

        Parameters
        ----------
        fcst : pd.DataFrame
            Forecasts from individual ensemble members.
            Expected columns: ['unique_id', 'ds', 'model_name1', 'model_name2', ...]

        Returns
        -------
        pd.Series
            Combined ensemble forecasts for h periods ahead.

        """

        self._assert_fcst(fcst)

        fcst_c = fcst.apply(lambda x: self._weighted_average(x, self.weights), axis=1)
        fcst_c.name = self.alias

        return fcst_c

    # pylint: disable=arguments-differ
    def update_weights(self, **kwargs):
        """Updating the combination weights


        Not implemented yet

        """

        raise NotImplementedError

    def _weights_by_uid(self, **kwargs):
        if self.weight_by_uid:
            top_models = self.insample_scores.apply(self._get_top_k, axis=1)
        else:
            top_models = self._get_top_k(self.insample_scores.mean())

        uid_weights = {}
        for uid, uid_scr in self.insample_scores.iterrows():
            weights = self._weights_from_errors(uid_scr)

            if self.weight_by_uid:
                poor_models = [x not in top_models[uid] for x in weights.index]
            else:
                poor_models = [x not in top_models for x in weights.index]

            weights[poor_models] = 0
            weights /= weights.sum()

            uid_weights[uid] = weights

        weights_df = pd.DataFrame(uid_weights).T
        weights_df.index.name = "unique_id"

        return weights_df
