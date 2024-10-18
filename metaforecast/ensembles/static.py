import pandas as pd

from metaforecast.ensembles.windowing import Windowing


class BestOnTrain(Windowing):
    """ BestOnTrain

    Selecting the ensemble member with the best performance on training data

   Example usage (CHECK NOTEBOOKS FOR MORE EXAMPLES)
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
        """
        :param select_by_uid: whether to select the best ensemble member by unique_id (True) or
        across all dataset (False)
        :type select_by_uid: bool
        """

        super().__init__(freq='',
                         select_best=True,
                         trim_ratio=1,
                         weight_by_uid=select_by_uid)

        self.alias = 'BestOnTrain'

        self.use_window = False
        self.select_by_uid = select_by_uid

    def update_weights(self, **kwargs):
        raise NotImplementedError


class LossOnTrain(Windowing):
    """ LossOnTrain

    Weighting the ensemble members according to the squared error on training data

    Example usage (CHECK NOTEBOOKS FOR MORE EXAMPLES)
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
        """

        :param weight_by_uid: Whether to weight the ensemble by unique_id (True) or dataset (False)
        Defaults to True, but this can become computationally demanding for datasets with a large
        number of time series
        :type weight_by_uid: bool

        :param trim_ratio: Ratio (0-1) of ensemble members to keep in the ensemble.
        (1-trim_ratio) of models will not be used during inference based on validation accuracy.
        Defaults to 1, which means all ensemble members are used.
        :type trim_ratio: float

        """
        super().__init__(freq='',
                         select_best=False,
                         trim_ratio=trim_ratio,
                         weight_by_uid=weight_by_uid)

        self.alias = 'LossOnTrain'

        self.use_window = False

    def update_weights(self, **kwargs):
        raise NotImplementedError


class EqAverage(Windowing):
    """ EqAverage

    Combining ensemble members with a simple average after a preliminary trimming

    References:
        Jose, V. R. R., & Winkler, R. L. (2008). Simple robust averages of
        forecasts: Some empirical results. International journal of forecasting, 24(1), 163-169.

    Example usage (CHECK NOTEBOOKS FOR MORE EXAMPLES)
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
        """
        :param select_by_uid: Whether to trim the ensemble by unique_id (True) or dataset (False)
        :type select_by_uid: bool

        :param trim_ratio: Ratio (0-1) of ensemble members to keep in the ensemble.
        (1-trim_ratio) of models will not be used during inference based on validation accuracy.
        Defaults to 1, which means all ensemble members are used.
        :type trim_ratio: float

        """
        super().__init__(freq='',
                         select_best=False,
                         trim_ratio=trim_ratio,
                         weight_by_uid=select_by_uid)

        self.alias = 'EqAverage'

        self.use_window = False

    def update_weights(self, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _weights_from_errors(scores: pd.Series) -> pd.Series:
        weights = pd.Series({k: 1 / len(scores) for k in scores.index})

        return weights
