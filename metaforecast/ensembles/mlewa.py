import typing

import pandas as pd
import numpy as np

from metaforecast.ensembles.base import Mixture

RowIDType = typing.Union[int, typing.Hashable]


class MLewa(Mixture):

    def __init__(self,
                 loss_type: str,
                 gradient: bool,
                 trim_ratio: float,
                 weight_by_uid: bool):

        self.alias = 'MLewa'

        super().__init__(loss_type=loss_type,
                         gradient=gradient,
                         trim_ratio=trim_ratio,
                         weight_by_uid=weight_by_uid)

    def _update_mixture(self, fcst: pd.DataFrame, y: np.ndarray, **kwargs):
        for i, fc in fcst.iterrows():
            w = self._weights_from_regret(iteration=i)

            self.weights[i], self.ensemble_fcst[i] = self._calc_ensemble_fcst(fc, w)

            loss_experts = self._calc_loss(fcst=fc, y=y[i], fcst_c=self.ensemble_fcst[i])
            loss_mixture = self._calc_loss(fcst=self.ensemble_fcst[i], y=y[i], fcst_c=self.ensemble_fcst[i])

            regret_i = (loss_mixture - loss_experts)

            # update regret
            for mod in self.regret:
                self.regret[mod] += regret_i[mod]

            n = len(self.model_names)
            self.eta[int(str(i)) + 1] = np.sqrt(np.log(n) / (np.log(n) / self.eta[i] ** 2 + regret_i ** 2))

    def _weights_from_regret(self, iteration: RowIDType = -1):
        curr_regret = np.array(list(self.regret.values()))

        if np.max(curr_regret) > 0:
            w = self.truncate_loss(np.exp(self.eta[iteration] * curr_regret))
            w /= np.sum(w)
        else:
            w = np.ones_like(curr_regret) / len(curr_regret)

        w = pd.Series(w, index=self.model_names)

        return w

    def _initialize_params(self, fcst: pd.DataFrame):

        n_row, n_col = fcst.shape[0], len(self.model_names)

        self.eta = np.full(shape=(n_row + 1, n_col), fill_value=np.exp(350))
        self.regret = {k: 0 for k in self.model_names}
        self.weights = np.zeros((n_row, n_col))
        self.ensemble_fcst = np.zeros(n_row)

    def update_weights(self, fcst: pd.DataFrame):
        raise NotImplementedError

    @staticmethod
    def truncate_loss(x):
        return np.clip(x, np.exp(-700), np.exp(700))
