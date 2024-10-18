from typing import Optional

import pandas as pd

from metaforecast.ensembles.base import ForecastingEnsemble


class Windowing(ForecastingEnsemble):
    """ Windowing

    Forecast combination based on windowing - forecast accuracy (squared error) on a
    recent window of data

    References:
        Cerqueira, V., Torgo, L., Oliveira, M., & Pfahringer, B. (2017, October).
        Dynamic and heterogeneous ensembles for time series forecasting. In 2017 IEEE international
        conference on data science and advanced analytics (DSAA) (pp. 242-251). IEEE.

        van Rijn, J. N., Holmes, G., Pfahringer, B., & Vanschoren, J. (2015,
        November). Having a blast: Meta-learning and heterogeneous ensembles
        for data streams. In 2015 ieee international conference on
        data mining (pp. 1003-1008). IEEE.

    Example usage (CHECK NOTEBOOKS FOR MORE EXAMPLES)
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

    def __init__(self,
                 freq: str,
                 select_best: bool = False,
                 trim_ratio: float = 1,
                 weight_by_uid: bool = False,
                 window_size: Optional[int] = None):
        """
        :param freq: Sampling frequency of the time series (e.g. 'M')
        :type freq: str

        :param select_best: Whether to select the single model that maximizes forecast performance
        on in-sample data
        :type select_best: bool

        :param trim_ratio: Ratio (0-1) of ensemble members to keep in the ensemble.
        (1-trim_ratio) of models will not be used during inference based on validation accuracy.
        Defaults to 1, which means all ensemble members are used.
        :type trim_ratio: float

        :param weight_by_uid: Whether to weight the ensemble by unique_id (True) or dataset (False)
        Defaults to True, but this can become computationally demanding for datasets with a large
        number of time series
        :type weight_by_uid: bool

        :param window_size: No of recent observations used to trim ensemble. If None, a size
        equivalent to the sampling frequency will be used.
        :type window_size: int
        """

        super().__init__()

        self.alias = 'Windowing'
        self.frequency = freq

        if window_size is None:
            self.window_size = self.WINDOW_SIZE_BY_FREQ[self.frequency]
        else:
            self.window_size = window_size

        self.select_best = select_best
        if self.select_best:
            self.trim_ratio = 1e-10
            self.alias = 'BLAST'
        else:
            self.trim_ratio = trim_ratio

        self.weight_by_uid = weight_by_uid
        self.insample_scores = None
        self.use_window = True

        self.weights = None

    def fit(self, insample_fcst, **kwargs):
        if self.model_names is None:
            self.model_names = insample_fcst.columns.to_list()
            self.model_names = [x for x in self.model_names if x not in self.METADATA + ['h']]

        self._set_n_models()

        self.insample_scores = self.evaluate_base_fcst(insample_fcst=insample_fcst,
                                                       use_window=self.use_window)

        self.weights = self._weights_by_uid()

    def predict(self, fcst: pd.DataFrame, **kwargs):
        self._assert_fcst(fcst)

        fcst_c = fcst.apply(lambda x: self._weighted_average(x, self.weights), axis=1)
        fcst_c.name = self.alias

        return fcst_c

    def update_weights(self, **kwargs):
        """ update_weights

        Updating loss statistics for dynamic model selection

        """

        raise NotImplementedError

    def _weights_by_uid(self):
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
        weights_df.index.name = 'unique_id'

        return weights_df
