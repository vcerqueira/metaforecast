import pandas as pd

from metaforecast.ensembles.windowing import Windowing


class BestOnTrain(Windowing):
    """Select single best model based on training performance.

    A simple baseline ensemble that selects the single best-performing model
    based on training data accuracy. While not technically an ensemble since
    it uses only one model, it serves as an important baseline.

    Examples
    --------
    >>> from datasetsforecast.m3 import M3
    >>> from neuralforecast import NeuralForecast
    >>> from neuralforecast.models import NHITS, NBEATS, MLP
    >>> from metaforecast.ensembles import BestOnTrain
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
    >>> ensemble = BestOnTrain()
    >>> ensemble.fit(fcst_cv)
    >>>
    >>> # re-fitting models
    >>> nf.fit(df=df)
    >>>
    >>> # forecasting and combining
    >>> fcst = nf.predict()
    >>> fcst_ensemble = ensemble.predict(fcst.reset_index())
    """

    def __init__(self, select_by_uid: bool = True):
        """Initialize best model selector.

        Parameters
        ----------
        select_by_uid : bool, default=True
            Strategy for selecting best performing model:
            - True: Select best model separately for each series
            - False: Select single best model across all series

        Notes
        -----
        Per-series selection (select_by_uid=True) allows for more granular model
        choice but requires sufficient data per series for reliable selection.
        Global selection may be more robust when individual series are short.

        """

        super().__init__(
            freq="",
            select_best=True,
            trim_ratio=1,
            weight_by_uid=select_by_uid,
        )

        self.alias = "BestOnTrain"

        self.use_window = False
        self.select_by_uid = select_by_uid

    def update_weights(self, **kwargs):
        raise NotImplementedError


class LossOnTrain(Windowing):
    """Weight ensemble members based on training set performance.

    An ensemble method that assigns static weights to models
    based on their training error. Unlike dynamic ensembles, weights
    are fixed after training and don't adapt to changing patterns.

    Notes
    -----
    Weights are computed as inverse of training error, giving higher
    weights to more accurate models. This static weighting assumes
    relative model performance remains stable over time.

    Examples
    --------
    >>> from datasetsforecast.m3 import M3
    >>> from neuralforecast import NeuralForecast
    >>> from neuralforecast.models import NHITS, NBEATS, MLP
    >>> from metaforecast.ensembles import LossOnTrain
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
    >>> ensemble = LossOnTrain(trim_ratio=0.8)
    >>> ensemble.fit(fcst_cv)
    >>>
    >>> # re-fitting models
    >>> nf.fit(df=df)
    >>>
    >>> # forecasting and combining
    >>> fcst = nf.predict()
    >>> fcst_ensemble = ensemble.predict(fcst.reset_index())

    """

    def __init__(self, trim_ratio: float, weight_by_uid: bool = True):
        """Initialize static ensemble with training-based weights.

        Parameters
        ----------
        weight_by_uid : bool, default=True
            Strategy for computing model weights:
            - True: Separate weights per series
            - False: Global weights across all series

        trim_ratio : float, default=1.0
            Proportion of models to retain in ensemble, between 0 and 1:
            - 1.0: Keep all models
            - 0.5: Keep top 50% of models
            Models are selected based on training performance

        Notes
        -----
        Weight computation involves:
        1. Calculate training error for each model
        2. Convert errors to weights (inverse relationship)
        3. If trim_ratio < 1, select top performing models
        4. Normalize weights to sum to 1

        """
        super().__init__(
            freq="",
            select_best=False,
            trim_ratio=trim_ratio,
            weight_by_uid=weight_by_uid,
        )

        self.alias = "LossOnTrain"

        self.use_window = False

    def update_weights(self, **kwargs):
        raise NotImplementedError


class EqAverage(Windowing):
    """Combine forecasts using simple average with optional trimming.

    A robust ensemble method that equally weights retained models after
    removing poor performers. Research shows this simple approach often
    performs competitively with more complex weighting schemes.

    References
    ----------
    Jose, V. R. R., & Winkler, R. L. (2008).
    "Simple robust averages of forecasts: Some empirical results."
    International Journal of Forecasting, 24(1), 163-169.

    Examples
    --------
    >>> from datasetsforecast.m3 import M3
    >>> from neuralforecast import NeuralForecast
    >>> from neuralforecast.models import NHITS, NBEATS, MLP
    >>> from metaforecast.ensembles import EqAverage
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
    >>> ensemble =  EqAverage()
    >>> ensemble.fit(fcst_cv)
    >>>
    >>> # re-fitting models
    >>> nf.fit(df=df)
    >>>
    >>> # forecasting and combining
    >>> fcst = nf.predict()
    >>> fcst_ensemble = ensemble.predict(fcst.reset_index())
    """

    def __init__(self, trim_ratio: float = 1, select_by_uid: bool = True):
        """Initialize equal-weights ensemble with optional trimming.

        Parameters
        ----------
        select_by_uid : bool, default=True
            Strategy for model selection in trimming:
            - True: Select best models separately for each series
            - False: Select best models across all series
            Per-series selection allows more granular model choice
            but requires sufficient data per series.

        trim_ratio : float, default=1.0
            Proportion of models to retain in ensemble, between 0 and 1:
            - 1.0: Keep all models (simple average)
            - 0.5: Keep top 50% of models
            - Lower values create more selective ensembles

        Notes
        -----
        Models are selected based on validation performance before
        applying equal weights. As shown in [1], moderate trimming
        often improves forecast accuracy while maintaining the
        robustness benefits of equal weighting.

        References
        ----------
        [1] Jose, V. R. R., & Winkler, R. L. (2008).
        "Simple robust averages of forecasts: Some empirical results."
        International Journal of Forecasting, 24(1), 163-169.
        """
        super().__init__(
            freq="",
            select_best=False,
            trim_ratio=trim_ratio,
            weight_by_uid=select_by_uid,
        )

        self.alias = "EqAverage"

        self.use_window = False

    def update_weights(self, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _weights_from_errors(scores: pd.Series) -> pd.Series:
        weights = pd.Series({k: 1 / len(scores) for k in scores.index})

        return weights
