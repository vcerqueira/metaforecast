"""Bernstein Online Aggregation (BOA) ensemble for forecast combination.

BOA uses a Bernstein-type inequality to adapt the learning rate, yielding
tighter regret bounds than standard exponential-weight methods.  It
maintains a *regularised* cumulative regret that penalises high-variance
experts, producing more stable weight trajectories.

Based on the implementation in the opera R package [1]_ and the theoretical
framework of Wintenberger (2017) [2]_.

References
----------
.. [1] Gaillard, P. & Goude, Y. (2015). "Forecasting electricity
   consumption by aggregating experts." In *Modeling and Stochastic Learning
   for Forecasting in High Dimensions* (pp. 95-115). Springer.

.. [2] Wintenberger, O. (2017). "Optimal learning with Bernstein online
   aggregation." *Machine Learning*, 106(1), 119-141.
"""

from __future__ import annotations

import typing

import numpy as np
import pandas as pd

from metaforecast.ensembles.base import Mixture

RowIdentifierType = int | typing.Hashable


class BOA(Mixture):
    """Online ensemble using Bernstein Online Aggregation.

    At each time step the algorithm:

    1. Computes weights from the regularised regret (Bernstein-style).
    2. Forms the mixture forecast.
    3. Observes losses and computes instantaneous regret.
    4. Adapts the per-expert learning rate via
       ``eta_inv2[t+1] = eta_inv2[t] + 2.2 * r^2``.
    5. Updates the regularised regret:
       ``r_reg = r - r^2 / sqrt(eta_inv2[t+1])``.

    The regularisation term ``-r^2 / sqrt(eta_inv2)`` penalises experts
    with high variance, improving concentration bounds.

    Parameters
    ----------
    loss_type : {'square', 'pinball', 'percentage', 'absolute', 'log'}
        Loss function for evaluating and weighting ensemble members.
    gradient : bool
        If True, use the gradient of the loss for weight updates.
    trim_ratio : float, default 1.0
        Proportion of models to retain (1.0 keeps all).
    weight_by_uid : bool, default False
        If True, maintain separate weights per series.

    Examples
    --------
    >>> from metaforecast.ensembles import BOA
    >>> ensemble = BOA(loss_type='square', gradient=True)
    >>> ensemble.fit(fcst_cv)
    >>> combined = ensemble.predict(fcst)

    See Also
    --------
    MLewa : Exponentially weighted averaging (simpler, fixed-rate).
    MLpol : Polynomially weighted averaging.
    """

    def __init__(
        self,
        loss_type: str,
        gradient: bool,
        trim_ratio: float = 1.0,
        weight_by_uid: bool = False,
    ):
        super().__init__(
            loss_type=loss_type,
            gradient=gradient,
            trim_ratio=trim_ratio,
            weight_by_uid=weight_by_uid,
        )
        self.alias = "BOA"

        self._eta_inv2: np.ndarray | None = None
        self._w0: np.ndarray | None = None
        self._R: np.ndarray | None = None
        self._R_reg: np.ndarray | None = None

    def _initialize_params(self, fcst: pd.DataFrame):
        n_row, n_col = fcst.shape[0], len(self.model_names)

        self._eta_inv2 = np.zeros((n_row + 1, n_col))
        self._w0 = np.ones(n_col)
        self._R = np.zeros(n_col)
        self._R_reg = np.zeros(n_col)

        self.regret = {k: 0.0 for k in self.model_names}
        self.weights = np.zeros((n_row, n_col))
        self.ensemble_fcst = np.zeros(n_row)

    def _update_mixture(self, fcst: pd.DataFrame, y: np.ndarray, **kwargs):
        for i, fc in fcst.iterrows():
            idx = int(str(i))

            w = self._weights_from_regret(iteration=idx)
            self.weights[idx], self.ensemble_fcst[idx] = self._calc_ensemble_fcst(fc, w)

            loss_experts = self._calc_loss(fcst=fc, y=y[idx], fcst_c=self.ensemble_fcst[idx])
            loss_mixture = self._calc_loss(
                fcst=self.ensemble_fcst[idx],
                y=y[idx],
                fcst_c=self.ensemble_fcst[idx],
            )

            r = (loss_mixture - loss_experts).values

            # Adapt per-expert learning rate
            self._eta_inv2[idx + 1] = self._eta_inv2[idx] + 2.2 * r**2

            # Regularised regret update
            nz = self._eta_inv2[idx + 1] > 0
            r_reg = np.zeros_like(r)
            if np.any(nz):
                r_reg[nz] = r[nz] - r[nz] ** 2 / np.sqrt(self._eta_inv2[idx + 1, nz])

            self._R += r
            self._R_reg += r_reg

            for j, mod in enumerate(self.regret):
                self.regret[mod] = self._R[j]

    def _weights_from_regret(self, iteration: RowIdentifierType = -1, **kwargs):
        w0 = self._w0
        ei2 = self._eta_inv2[iteration]
        nz = ei2 > 0

        w = w0.copy()
        if np.any(nz):
            R_aux = -np.log(ei2[nz]) / 2 + np.log(w0[nz]) + self._R_reg[nz] / np.sqrt(ei2[nz])
            R_max = np.max(R_aux)
            exp_aux = np.exp(R_aux - R_max)
            w[nz] = np.sum(w0[nz]) * exp_aux / np.sum(exp_aux)

        w_sum = np.sum(w)
        w = w / w_sum if w_sum > 0 else np.ones(len(self.model_names)) / len(self.model_names)

        return pd.Series(w, index=self.model_names)

    def update_weights(self, fcst: pd.DataFrame):
        raise NotImplementedError
