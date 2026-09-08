"""MLprod (Multiplicative Weights with Production Update) ensemble.

MLprod combines expert forecasts using a multiplicative update in log-space.
The learning rate is simultaneously bounded by the inverse of the maximum
observed regret (stability) and the square root of log(N)/L (optimal rate),
and the cumulative regret is rescaled at each step to match the new rate.

Based on the implementation in the opera R package [1]_.

References
----------
.. [1] Gaillard, P. & Goude, Y. (2015). "Forecasting electricity
   consumption by aggregating experts." In *Modeling and Stochastic Learning
   for Forecasting in High Dimensions* (pp. 95-115). Springer.

.. [2] Cesa-Bianchi, N. & Lugosi, G. (2006). *Prediction, Learning, and
   Games.* Cambridge University Press.
"""

from __future__ import annotations

import typing

import numpy as np
import pandas as pd

from metaforecast.ensembles.base import Mixture

RowIdentifierType = typing.Union[int, typing.Hashable]


class MLprod(Mixture):
    """Online ensemble using the MLprod multiplicative update.

    At each time step the algorithm:

    1. Computes weights ``w = eta * exp(R) / sum(eta * exp(R))``.
    2. Forms the mixture forecast.
    3. Observes losses and computes instantaneous regret ``r``.
    4. Updates cumulative squared regret ``L`` and ``maxloss``.
    5. Adapts the learning rate:
       ``eta = min(1/(2*maxloss), sqrt(log(N)/L))``.
    6. Rescales and updates the log-regret:
       ``R = (eta_new / eta_old) * R + log(1 + eta_new * r)``.

    The rescaling in step 6 is the distinctive feature: it allows the
    algorithm to "forget" early observations as the learning rate decreases,
    achieving near-optimal regret in non-stationary settings.

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
    >>> from metaforecast.ensembles import MLprod
    >>> ensemble = MLprod(loss_type='square', gradient=True)
    >>> ensemble.fit(fcst_cv)
    >>> combined = ensemble.predict(fcst)

    See Also
    --------
    MLpol : Polynomially weighted averaging (additive update).
    MLewa : Exponentially weighted averaging.
    BOA : Bernstein Online Aggregation.
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
        self.alias = "MLprod"

        self._R: np.ndarray | None = None
        self._L: np.ndarray | None = None
        self._maxloss: np.ndarray | None = None

    def _initialize_params(self, fcst: pd.DataFrame):
        n_row, n_col = fcst.shape[0], len(self.model_names)

        self._R = np.zeros(n_col)  # log(w0) = log(1) = 0
        self._L = np.ones(n_col)
        self._maxloss = np.zeros(n_col)

        self.eta = np.full((n_row + 1, n_col), np.exp(700))
        self.regret = {k: 0.0 for k in self.model_names}
        self.weights = np.zeros((n_row, n_col))
        self.ensemble_fcst = np.zeros(n_row)

    def _update_mixture(self, fcst: pd.DataFrame, y: np.ndarray, **kwargs):
        n_experts = len(self.model_names)

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

            # Update cumulative statistics
            self._L += r**2
            self._maxloss = np.maximum(self._maxloss, np.abs(r))

            # Adapt learning rate (element-wise)
            with np.errstate(divide="ignore"):
                term_stability = 1.0 / (2.0 * self._maxloss)
            term_optimal = np.sqrt(np.log(n_experts) / self._L)
            neweta = np.minimum(term_stability, term_optimal)

            # Rescale and update log-regret
            self._R = (neweta / self.eta[idx]) * self._R + np.log1p(neweta * r)
            self.eta[idx + 1] = neweta

            for j, mod in enumerate(self.regret):
                self.regret[mod] = self._R[j]

    def _weights_from_regret(self, iteration: RowIdentifierType = -1, **kwargs):
        et = self.eta[iteration]
        R_max = np.max(self._R)

        w = np.exp(self._R - R_max)
        w = et * w
        w_sum = np.sum(w)

        w = w / w_sum if w_sum > 0 else np.ones(len(self.model_names)) / len(self.model_names)

        return pd.Series(w, index=self.model_names)

    def update_weights(self, fcst: pd.DataFrame):
        raise NotImplementedError
