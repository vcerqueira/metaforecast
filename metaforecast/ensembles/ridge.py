"""Online Ridge regression ensemble for forecast combination.

The Ridge ensemble computes combination weights via online ridge regression.
At each time step the algorithm solves a regularised least-squares problem
to find the weight vector that best reconstructs the target using the
expert predictions seen so far.

Based on the ridge regression aggregation rule in the opera R package [1]_.

References
----------
.. [1] Gaillard, P. & Goude, Y. (2015). "Forecasting electricity
   consumption by aggregating experts." In *Modeling and Stochastic Learning
   for Forecasting in High Dimensions* (pp. 95-115). Springer.

.. [2] Cesa-Bianchi, N. & Lugosi, G. (2006). *Prediction, Learning, and
   Games.* Cambridge University Press.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import linalg as la

from metaforecast.ensembles.base import Mixture


class Ridge(Mixture):
    """Online ensemble using ridge regression weight updates.

    At each time step the algorithm:

    1. Solves ``w = A_t^{-1} b_t`` for the current weight vector.
    2. Forms the mixture forecast using ``w``.
    3. Updates the sufficient statistics:
       ``A_t <- A_t + x_t x_t^T``, ``b_t <- b_t + x_t y_t``,
       where ``x_t`` is the vector of expert predictions and ``y_t`` the
       observation.

    The regularisation parameter ``lambda_`` controls the strength of the
    L2 penalty (``lambda_ * I`` initialises ``A_0``).

    Parameters
    ----------
    lambda_ : float, default 1.0
        Ridge regularisation strength (> 0).  Larger values shrink the
        weights toward zero / equal weighting.
    loss_type : str, default 'square'
        Loss type (kept for API consistency with ``Mixture``; the Ridge
        weight update itself always minimises squared error).
    gradient : bool, default False
        Passed through to ``Mixture`` (has no effect on the ridge update).
    trim_ratio : float, default 1.0
        Proportion of models to retain (1.0 keeps all).
    weight_by_uid : bool, default False
        If True, maintain separate weights per series.

    Examples
    --------
    >>> from metaforecast.ensembles import Ridge
    >>> ensemble = Ridge(lambda_=1.0)
    >>> ensemble.fit(fcst_cv)
    >>> combined = ensemble.predict(fcst)

    See Also
    --------
    MLpol : Polynomially weighted averaging.
    MLewa : Exponentially weighted averaging.
    FixedShare : Fixed-share weight updates with sharing parameter.
    """

    def __init__(
        self,
        lambda_: float = 1.0,
        loss_type: str = "square",
        gradient: bool = False,
        trim_ratio: float = 1.0,
        weight_by_uid: bool = False,
    ):
        super().__init__(
            loss_type=loss_type,
            gradient=gradient,
            trim_ratio=trim_ratio,
            weight_by_uid=weight_by_uid,
        )

        self.alias = "Ridge"
        self.lambda_ = lambda_

        self._At: np.ndarray | None = None
        self._bt: np.ndarray | None = None

    def _initialize_params(self, fcst: pd.DataFrame):
        n_row, n_col = fcst.shape[0], len(self.model_names)

        self._At = self.lambda_ * np.eye(n_col)
        self._bt = np.zeros(n_col)

        self.weights = np.zeros((n_row, n_col))
        self.ensemble_fcst = np.zeros(n_row)
        self.regret = {k: 0.0 for k in self.model_names}

    def _update_mixture(self, fcst: pd.DataFrame, y: np.ndarray, **kwargs):
        for i, fc in fcst.iterrows():
            w = self._weights_from_regret()

            self.weights[i], self.ensemble_fcst[i] = self._calc_ensemble_fcst(fc, w)

            x = fc.values.astype(np.float64)
            self._At = self._At + np.outer(x, x)
            self._bt = self._bt + x * y[i]

    def _weights_from_regret(self, **kwargs):
        w = la.solve(self._At, self._bt, assume_a="pos")

        return pd.Series(w, index=self.model_names)

    def update_weights(self, fcst: pd.DataFrame):
        raise NotImplementedError
