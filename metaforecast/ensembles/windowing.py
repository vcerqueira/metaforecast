from typing import Optional

import pandas as pd

from metaforecast.ensembles.base import ForecastingEnsemble


class Windowing(ForecastingEnsemble):
    """ Windowing

    Forecast combination based on windowing - forecast accuracy (squared error) on a recent window of data

    """

    def __init__(self,
                 freq: str,
                 select_best: bool,
                 trim_ratio: float = 1,
                 weight_by_uid: bool = False,
                 window_size: Optional[int] = None):
        """
        :param trim_ratio:


        :param window_size: No of recent observations used to trim ensemble
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
