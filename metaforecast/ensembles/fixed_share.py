"""Fixed-Share ensemble for online forecast combination.

The Fixed-Share algorithm is a regret-minimization method that maintains a
pool of expert weights and, at each round, "shares" a fraction alpha of the
total weight uniformly across all experts.  This prevents the weights from
collapsing onto a single model and makes the algorithm robust to abrupt
regime changes.

Based on the algorithm described in Herbster & Warmuth (1998) [1]_ and the
implementation in the opera R package [2]_.

References
----------
.. [1] Herbster, M. & Warmuth, M. K. (1998). "Tracking the Best Expert."
   *Machine Learning*, 32(2), 151-178.

.. [2] Gaillard, P. & Goude, Y. (2015). "Forecasting electricity
   consumption by aggregating experts." In *Modeling and Stochastic Learning
   for Forecasting in High Dimensions* (pp. 95-115). Springer.
"""

from __future__ import annotations

import typing

import numpy as np
import pandas as pd

from metaforecast.ensembles.base import EXPERT_LOSS, Mixture

RowIdentifierType = typing.Union[int, typing.Hashable]


class FixedShare(Mixture):
    """Online ensemble using Fixed-Share weight updates.

    At each time step the algorithm:

    1. Forms the mixture forecast using current weights.
    2. Observes the loss of every expert and of the mixture.
    3. Updates the regret of each expert.
    4. Applies a *sharing* step: a fraction ``alpha`` of each expert's
       exponentiated regret leaks uniformly to all experts, preventing
       any single expert from permanently dominating.

    The sharing step makes Fixed-Share particularly suited to settings
    where the best expert changes over time (e.g. concept drift).

    Parameters
    ----------
    loss_type : {'square', 'pinball', 'percentage', 'absolute', 'log'}
        Loss function for evaluating and weighting ensemble members.
    gradient : bool
        If True, use the gradient of the loss for weight updates.
    eta : float
        Learning rate (> 0).  Controls the sensitivity of weights to
        cumulative regret.
    alpha : float
        Sharing parameter in (0, 1).  At each round, a fraction ``alpha``
        of each expert's weight is redistributed uniformly.  Larger values
        make the algorithm more reactive to regime changes.
    trim_ratio : float, default 1.0
        Proportion of models to retain (1.0 keeps all).
    weight_by_uid : bool, default False
        If True, maintain separate weights per series.

    Examples
    --------
    >>> from metaforecast.ensembles import FixedShare
    >>> ensemble = FixedShare(
    ...     loss_type='square', gradient=True, eta=0.1, alpha=0.01,
    ... )
    >>> ensemble.fit(fcst_cv)
    >>> combined = ensemble.predict(fcst)

    See Also
    --------
    MLewa : Exponentially weighted averaging (no sharing step).
    MLpol : Polynomially weighted averaging.
    """

    def __init__(
        self,
        loss_type: str,
        gradient: bool,
        eta: float,
        alpha: float,
        trim_ratio: float = 1.0,
        weight_by_uid: bool = False,
    ):
        super().__init__(
            loss_type=loss_type,
            gradient=gradient,
            trim_ratio=trim_ratio,
            weight_by_uid=weight_by_uid,
        )

        self.alias = "FixedShare"
        self.eta = eta
        self.alpha = alpha
        self.cum_loss: float = 0.0

    def _initialize_params(self, fcst: pd.DataFrame):
        n_row, n_col = fcst.shape[0], len(self.model_names)

        self.w0 = {k: 1.0 for k in self.model_names}
        self.regret = {k: np.log(v) / self.eta for k, v in self.w0.items()}

        self.weights = np.zeros((n_row, n_col))
        self.ensemble_fcst = np.zeros(n_row)
        self.cum_loss = 0.0

    def _update_mixture(self, fcst: pd.DataFrame, y: np.ndarray, **kwargs):
        n_models = len(self.model_names)

        for i, fc in fcst.iterrows():
            w = self._weights_from_regret()

            self.weights[i], self.ensemble_fcst[i] = self._calc_ensemble_fcst(fc, w)

            self.cum_loss += EXPERT_LOSS[self.loss_type].loss(fcst=self.ensemble_fcst[i], y=y[i])

            loss_experts = self._calc_loss(fcst=fc, y=y[i], fcst_c=self.ensemble_fcst[i])
            loss_mixture = self._calc_loss(
                fcst=self.ensemble_fcst[i],
                y=y[i],
                fcst_c=self.ensemble_fcst[i],
            )

            regret_i = loss_mixture - loss_experts

            for mod in self.regret:
                reg_updated = self.regret[mod] + regret_i[mod]
                exp_reg = _truncate(np.exp(self.eta * reg_updated))
                self.regret[mod] = (
                    np.log(self.alpha / n_models + (1 - self.alpha) * exp_reg) / self.eta
                )

    def _weights_from_regret(self, **kwargs):
        curr_regret = np.array(list(self.regret.values()))

        w = _truncate(np.exp(self.eta * curr_regret))
        w = w / np.sum(w)

        return pd.Series(w, index=self.model_names)

    def update_weights(self, fcst: pd.DataFrame):
        raise NotImplementedError


def _truncate(x: np.ndarray) -> np.ndarray:
    """Clip extreme exponential values to avoid overflow."""
    return np.clip(x, np.exp(-700), np.exp(700))
