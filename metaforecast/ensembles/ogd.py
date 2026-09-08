"""Online Gradient Descent (OGD) ensemble for forecast combination.

OGD performs gradient descent on the mixture loss at each time step, with an
optionally decaying learning rate and simplex projection to ensure valid
convex combination weights.

Based on the implementation in the opera R package [1]_ and the standard
online convex optimisation framework of Zinkevich (2003) [2]_.

References
----------
.. [1] Gaillard, P. & Goude, Y. (2015). "Forecasting electricity
   consumption by aggregating experts." In *Modeling and Stochastic Learning
   for Forecasting in High Dimensions* (pp. 95-115). Springer.

.. [2] Zinkevich, M. (2003). "Online convex programming and generalized
   infinitesimal gradient ascent." In *ICML* (pp. 928-935).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from metaforecast.ensembles.base import Mixture


def _simplex_projection(v: np.ndarray) -> np.ndarray:
    """Project *v* onto the probability simplex {w >= 0, sum(w) = 1}.

    Uses the algorithm of Duchi et al. (2008).
    """
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho_candidates = np.nonzero(u * np.arange(1, n + 1) > cssv)[0]
    if len(rho_candidates) == 0:
        return np.ones(n) / n
    rho = rho_candidates[-1]
    theta = cssv[rho] / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


class OGD(Mixture):
    """Online ensemble using Online Gradient Descent.

    At each time step the algorithm:

    1. Predicts using the current weight vector.
    2. Observes the loss gradient of each expert.
    3. Tracks ``B``, the running maximum gradient norm.
    4. Computes the learning rate ``eta = t^{-alpha} / B``.
    5. Performs a gradient step: ``w <- w - eta * grad``.
    6. (Optionally) projects ``w`` onto the probability simplex.

    OGD is the simplest online convex optimisation baseline and serves as
    a useful reference point for more sophisticated methods.

    Parameters
    ----------
    alpha : float, default 0.5
        Learning-rate decay exponent in (0, 1].  ``eta_t = t^{-alpha} / B``.
        Smaller values yield slower decay (more aggressive updates).
    simplex : bool, default True
        If True, project weights onto the probability simplex after each
        gradient step, ensuring non-negative weights that sum to 1.
    loss_type : {'square', 'pinball', 'percentage', 'absolute', 'log'}
        Loss function (used to compute gradients).
    gradient : bool, default True
        Must be True for OGD (gradient-based updates).
    trim_ratio : float, default 1.0
        Proportion of models to retain (1.0 keeps all).
    weight_by_uid : bool, default False
        If True, maintain separate weights per series.

    Examples
    --------
    >>> from metaforecast.ensembles import OGD
    >>> ensemble = OGD(alpha=0.5)
    >>> ensemble.fit(fcst_cv)
    >>> combined = ensemble.predict(fcst)

    See Also
    --------
    MLpol : Polynomially weighted averaging (regret-based).
    Ridge : Online ridge regression.
    FixedShare : Fixed-share weight updates.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        simplex: bool = True,
        loss_type: str = "square",
        gradient: bool = True,
        trim_ratio: float = 1.0,
        weight_by_uid: bool = False,
    ):
        super().__init__(
            loss_type=loss_type,
            gradient=gradient,
            trim_ratio=trim_ratio,
            weight_by_uid=weight_by_uid,
        )
        self.alias = "OGD"
        self.alpha = alpha
        self.simplex = simplex

        self._w: np.ndarray | None = None
        self._B: float = 0.0
        self._t: int = 0

    def _initialize_params(self, fcst: pd.DataFrame):
        n_row, n_col = fcst.shape[0], len(self.model_names)

        self._w = np.ones(n_col) / n_col
        if self.simplex:
            self._w = _simplex_projection(self._w)
        self._B = 0.0
        self._t = 0

        self.regret = {k: 0.0 for k in self.model_names}
        self.weights = np.zeros((n_row, n_col))
        self.ensemble_fcst = np.zeros(n_row)

    def _update_mixture(self, fcst: pd.DataFrame, y: np.ndarray, **kwargs):
        for i, fc in fcst.iterrows():
            idx = int(str(i))
            self._t += 1

            w = pd.Series(self._w, index=self.model_names)
            self.weights[idx], self.ensemble_fcst[idx] = self._calc_ensemble_fcst(fc, w)

            # Compute gradient of the mixture loss w.r.t. the weights
            grad = self._calc_loss(fcst=fc, y=y[idx], fcst_c=self.ensemble_fcst[idx])
            grad_arr = grad.values if isinstance(grad, pd.Series) else np.asarray(grad)

            # Track maximum gradient norm
            self._B = max(self._B, np.sqrt(np.sum(grad_arr**2)))

            # Decaying learning rate
            eta = self._t ** (-self.alpha) / self._B if self._B > 0 else 1.0

            # Gradient step
            self._w = self._w - eta * grad_arr

            # Optional simplex projection
            if self.simplex:
                self._w = _simplex_projection(self._w)

    def _weights_from_regret(self, **kwargs):
        return pd.Series(self._w, index=self.model_names)

    def update_weights(self, fcst: pd.DataFrame):
        raise NotImplementedError
