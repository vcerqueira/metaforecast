"""PCA-based multi-label regressor for the MetaARIMA meta-learner.

Reduces the label space (ARIMA configuration scores) via PCA, trains a
multi-output regressor in the reduced space, and reconstructs predictions
back to the original label space.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import BaseEstimator, MultiOutputMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils.validation import check_array, check_X_y

warnings.filterwarnings("ignore")


class MultiLabelPCARegressor(BaseEstimator, MultiOutputMixin):
    """PCA transformation with multi-output regression.

    Combines:

    1. Feature standardisation (``StandardScaler``).
    2. Label-space dimensionality reduction via PCA.
    3. Multi-target regression in the reduced label space.
    4. Inverse PCA to recover predictions in the original label space.

    Parameters
    ----------
    mod : estimator
        A scikit-learn-compatible multi-output regressor (e.g.
        ``CatBoostRegressor``).  Must support ``.fit(X, y)`` and
        ``.predict(X)``.
    n_components : int, default 100
        Number of PCA components for the label space.  Capped at
        ``min(n_components, n_targets)`` during fit.
    random_state : int, default 1
        Random seed for PCA.
    """

    def __init__(self, mod, n_components: int = 100, random_state: int = 1):
        self.n_components = n_components
        self.random_state = random_state

        self.regressor = mod

        self.pca: PCA | None = None
        self.feature_scaler_ = StandardScaler()
        self.target_scaler_ = RobustScaler()
        self.n_features_in_: int | None = None
        self.n_outputs_: int | None = None

        self.is_fit: bool = False
        self.is_meta_regression: bool = False

    def fit(self, X, y, process_y: bool = False):
        """Fit the PCA regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples, n_targets)
            Target matrix (e.g. per-configuration error scores).
        process_y : bool, default False
            If True, winsorise and clean ``y`` before PCA (for
            meta-regression with noisy loss values).
        """
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)

        self.n_features_in_ = X.shape[1]
        self.n_outputs_ = y.shape[1]
        self.pca = PCA(
            n_components=min(self.n_components, y.shape[1]),
            random_state=self.random_state,
        )

        X_scaled = self.feature_scaler_.fit_transform(X)

        if process_y:
            self.is_meta_regression = True
            y_clean = np.where(np.isfinite(y), y, np.nan)
            for j in range(y_clean.shape[1]):
                finites = y_clean[:, j][np.isfinite(y_clean[:, j])]
                fill = np.percentile(finites, 99) if len(finites) > 0 else 0.0
                y_clean[:, j] = np.where(np.isnan(y_clean[:, j]), fill, y_clean[:, j])
            for j in range(y_clean.shape[1]):
                lo, hi = np.nanpercentile(y_clean[:, j], [1, 99])
                y_clean[:, j] = np.clip(y_clean[:, j], lo, hi)
            Z = self.pca.fit_transform(y_clean)
        else:
            Z = self.pca.fit_transform(y)

        self.regressor.fit(X_scaled, Z)
        self.is_fit = True

        return self

    def predict(self, X):
        """Predict in the original label space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        Y_pred : ndarray of shape (n_samples, n_targets)
        """
        X = check_array(X)
        if not self.is_fit:
            raise ValueError(
                "This model instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )

        X_scaled = self.feature_scaler_.transform(X)
        Z_pred = self.regressor.predict(X_scaled)
        return self.pca.inverse_transform(Z_pred)

    def predict_proba(self, X):
        """Alias for :meth:`predict` (returns PCA-reconstructed scores)."""
        return self.predict(X)
