"""MetaARIMA -- meta-learning for ARIMA configuration selection.

MetaARIMA is a two-stage algorithm:

1. **Meta-training** (``meta_fit``): given a feature matrix *X* (time-series
   features) and an error matrix *Y* (per-configuration scores), train a
   meta-learner that predicts which ARIMA configurations are likely to
   perform well for a new series.

2. **Inference** (``fit`` + ``predict``): for a new time series, extract
   features, query the meta-learner for a shortlist of promising
   configurations, fit them (optionally via successive halving), and
   select the best by AICc.

A pre-trained MetaARIMA can be serialised with :meth:`save` and restored
with :meth:`load`.

References
----------
.. [1] todo
"""

from __future__ import annotations

import copy
import gzip
import io
import sys
import warnings
from typing import List

import joblib
import numpy as np
import pandas as pd

from metaforecast.coseal._multilabel_pca import MultiLabelPCARegressor
from metaforecast.coseal.metaarima._base import (
    MetaARIMAUtils,
    _HalvingMetaARIMABase,
    _MetaARIMABase,
    tsfeatures_uid,
)

warnings.filterwarnings(action="ignore")


class MetaARIMA:
    """Meta-learning-based ARIMA configuration selector.

    Uses a multi-label meta-learner to shortlist ARIMA configurations,
    then selects the best configuration via AICc (exhaustive or
    successive-halving search).

    Parameters
    ----------
    model : estimator
        A scikit-learn-compatible regressor for the meta-learner (e.g.
        ``CatBoostRegressor``).
    freq : str
        Time-series frequency string used during meta-training (e.g.
        ``'ME'`` for monthly).
    season_length : int
        Seasonal period used during meta-training (e.g. 12 for monthly).
    n_trials : int
        Number of configurations to shortlist from the meta-learner.
    quantile_thr : float, default 0.5
        Quantile threshold for binarising error scores during
        meta-training: a configuration is labelled "good" if its error
        is at or below this quantile.
    pca_n_components : int, default 100
        Number of PCA components for the label space.
    mmr_lambda : float, default 0.75
        Maximal Marginal Relevance trade-off for the configuration
        shortlist.  ``1`` ranks by predicted score only (no diversity);
        ``0`` maximises diversity.  Values in ``(0, 1)`` interpolate
        between the two.
    base_optim : {'halving', 'complete'}, default 'halving'
        Strategy for selecting the final ARIMA from the shortlist:
        ``'halving'`` uses successive halving; ``'complete'`` fits all.

    Examples
    --------
    **Load a pre-trained model and run inference:**

    >>> from metaforecast.coseal import MetaARIMA
    >>> meta = MetaARIMA.load("trained_metaarima_m4_monthly.joblib.gz")
    >>> meta.fit(train_df, freq="ME", seas_length=12)
    >>> forecast = meta.predict(h=12)

    **Train from scratch:**

    >>> from catboost import CatBoostRegressor
    >>> from metaforecast.coseal import MetaARIMA
    >>> model = CatBoostRegressor(**params)
    >>> meta = MetaARIMA(model=model, freq="ME", season_length=12,
    ...                  n_trials=25)
    >>> meta.meta_fit(X_features, Y_scores)
    >>> meta.save("my_metaarima.joblib.gz")
    """

    BASE_OPTIM_METHODS = ["complete", "halving"]

    def __init__(
        self,
        model,
        freq: str,
        season_length: int,
        n_trials: int,
        quantile_thr: float = 0.5,
        pca_n_components: int = 100,
        mmr_lambda: float = 0.75,
        base_optim: str = "halving",
    ):
        assert base_optim in self.BASE_OPTIM_METHODS, (
            f"base_optim must be one of {self.BASE_OPTIM_METHODS}"
        )
        if not 0 <= mmr_lambda <= 1:
            raise ValueError("mmr_lambda must be in [0, 1].")

        self.n_trials = n_trials
        self.quantile_thr = quantile_thr
        self.model_names: list | None = None
        self.freq = freq
        self.freq_inference = freq
        self.season_length = season_length
        self.season_length_inference = season_length
        self.model = None
        self.corr_mat = None
        self.corr_mat_values = None
        self.pca_n_components = pca_n_components
        self.mmr_lambda = mmr_lambda
        self.base_optim = base_optim
        self.selected_config: str = ""

        self.meta_model = copy.deepcopy(
            MultiLabelPCARegressor(mod=model, n_components=self.pca_n_components)
        )

        self.is_fit: bool = False

    def meta_fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        """Train the meta-learner from pre-computed features and scores.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_series, n_features)
            Time-series features (e.g. from ``tsfeatures``).
        Y : pd.DataFrame of shape (n_series, n_configurations)
            Per-configuration error scores.  Columns must match the
            alias strings produced by
            :meth:`MetaARIMAUtils.get_models_sf`.
        """
        self.model_names = Y.columns.tolist()

        y = Y.apply(lambda x: (x <= x.quantile(self.quantile_thr)).astype(int), axis=1)

        if self.mmr_lambda < 1:
            self.corr_mat = Y.corr(method="kendall")
            self.corr_mat_values = self.corr_mat.values

        self.meta_model.fit(X, y)
        self.is_fit = True

    def meta_predict(self, X) -> list:
        """Predict the shortlist of promising ARIMA configurations.

        Parameters
        ----------
        X : array-like of shape (n_series, n_features)

        Returns
        -------
        list of list of str
            For each input series, a list of ARIMA alias strings.
        """
        assert self.is_fit, "Call meta_fit() before meta_predict()."

        meta_preds = self.meta_model.predict_proba(X)

        if isinstance(meta_preds, list):
            meta_preds = np.asarray([x[:, 1] for x in meta_preds]).T

        if self.mmr_lambda < 1:
            meta_preds_series = [pd.Series(x, index=self.model_names) for x in meta_preds]
            meta_preds_list = []
            for meta_pred in meta_preds_series:
                selected_indices = self._mmr_selection(probs=meta_pred.values)
                meta_preds_list.append(meta_pred.index[selected_indices].tolist())
        else:
            preds = pd.DataFrame(meta_preds, columns=self.model_names)
            meta_preds_list = preds.apply(
                lambda x: x.sort_values(ascending=False).index[: self.n_trials].tolist(),
                axis=1,
            ).values.tolist()

        return meta_preds_list

    def fit(self, df: pd.DataFrame, freq: str, seas_length: int):
        """Fit the best ARIMA to a new time series.

        Extracts features, queries the meta-learner for a shortlist,
        then selects the best configuration by AICc.

        Parameters
        ----------
        df : pd.DataFrame
            Single-series DataFrame with columns
            ``[unique_id, ds, y]``.
        freq : str
            Time-series frequency (e.g. ``'ME'``, ``'QE'``).
        seas_length : int
            Seasonal period of the target series.
        """
        self.freq_inference = freq
        self.season_length_inference = seas_length

        seas_fill = seas_length == 1 and self.season_length > 1

        feat_df = tsfeatures_uid(df, freq=self.season_length_inference, impute_seas=seas_fill)

        config_space = self.meta_predict(feat_df)[0]

        if self.season_length_inference != self.season_length:
            config_space = [
                s.replace(
                    f"[{self.season_length}]",
                    f"[{self.season_length_inference}]",
                )
                for s in config_space
            ]

        self._fit_on_configs(df, config_space)

    def predict(self, h: int, level: List | None = None) -> pd.DataFrame:
        """Forecast *h* steps ahead with the selected ARIMA.

        Parameters
        ----------
        h : int
            Forecast horizon.
        level : list of int, optional
            Prediction interval levels (e.g. ``[95]``).

        Returns
        -------
        pd.DataFrame
            Forecast DataFrame with ``unique_id``, ``ds``, and
            ``MetaARIMA`` columns (plus interval columns if ``level``
            is specified).
        """
        return self.model.predict(h, level=level)

    def save(self, path: str):
        """Save the fitted MetaARIMA to a gzip-compressed joblib file.

        Parameters
        ----------
        path : str
            Output file path (e.g. ``"metaarima_monthly.joblib.gz"``).
        """
        with gzip.open(path, "wb") as f:
            joblib.dump(self, f)

    @staticmethod
    def load(path: str) -> MetaARIMA:
        """Load a pre-trained MetaARIMA from disk.

        Handles models saved with either the current module paths or the
        legacy ``src.meta.arima.*`` paths from the experiments directory.

        Parameters
        ----------
        path : str
            Path to a ``.joblib.gz`` file produced by :meth:`save` (or
            the legacy ``ModelIO.save_model``).

        Returns
        -------
        MetaARIMA
        """
        with gzip.open(path, "rb") as f:
            data = f.read()

        buf = io.BytesIO(data)
        try:
            return joblib.load(buf)
        except ModuleNotFoundError:
            buf.seek(0)
            return _load_legacy(buf)

    def _fit_on_configs(self, df: pd.DataFrame, config_space: List[str]):
        assert self.is_fit

        base_params = {
            "config_space": config_space,
            "freq": self.freq_inference,
            "season_length": self.season_length_inference,
        }

        if self.base_optim == "complete":
            self.model = _MetaARIMABase(**base_params)
        elif self.base_optim == "halving":
            self.model = _HalvingMetaARIMABase(
                **base_params,
                eta=2,
                init_resource_factor=5,
                resource_factor=2,
            )
        else:
            raise ValueError(f"Unknown base optimizer: {self.base_optim}")

        self.model.fit(df)

        try:
            self.selected_config = MetaARIMAUtils.get_model_order(
                self.model.sf.fitted_[0][0].model_,
                as_alias=True,
                alias_freq=self.season_length,
            )
        except KeyError:
            self.selected_config = "MSTL"

    def _mmr_selection(self, probs: np.ndarray) -> list:
        """Re-rank configurations by Maximal Marginal Relevance.

        Balances predicted probability (relevance) and diversity (low
        correlation with already selected configurations).

        ``score = lambda * prob - (1-lambda) * max_corr``
        """
        n_configs = len(probs)
        selected_indices: list = []
        remaining_mask = np.ones(n_configs, dtype=bool)

        best_idx = int(np.argmax(probs))
        selected_indices.append(best_idx)
        remaining_mask[best_idx] = False

        lambda_probs = self.mmr_lambda * probs
        one_minus_lambda = 1 - self.mmr_lambda

        while len(selected_indices) < self.n_trials and remaining_mask.any():
            remaining_indices = np.where(remaining_mask)[0]
            if len(remaining_indices) == 0:
                break

            corr_matrix = self.corr_mat_values[np.ix_(remaining_indices, selected_indices)]
            max_corrs = np.max(corr_matrix, axis=1)

            mmr_scores = lambda_probs[remaining_indices] - one_minus_lambda * max_corrs

            best_relative_idx = int(np.argmax(mmr_scores))
            next_best_idx = remaining_indices[best_relative_idx]

            selected_indices.append(int(next_best_idx))
            remaining_mask[next_best_idx] = False

        return selected_indices


_MODULE_REMAP = {
    "src.meta.arima.meta_arima": "metaforecast.coseal.metaarima.meta_arima",
    "src.meta.arima._base": "metaforecast.coseal.metaarima._base",
    "src.meta.arima.multilabel_pca": "metaforecast.coseal._multilabel_pca",
}


def _load_legacy(buf: io.BytesIO):
    """Load a joblib-pickled object that used the old ``src.meta.arima.*`` paths.

    Temporarily installs the new modules under the old names in
    ``sys.modules`` so that joblib's NumpyUnpickler can resolve them.
    """
    import metaforecast.coseal._multilabel_pca as _mlpca
    import metaforecast.coseal.metaarima._base as _base
    import metaforecast.coseal.metaarima.meta_arima as _ma

    stubs: dict[str, object] = {}
    for old, new_mod in [
        ("src", sys.modules.get("metaforecast")),
        ("src.meta", sys.modules.get("metaforecast.coseal")),
        ("src.meta.arima", sys.modules.get("metaforecast.coseal.metaarima")),
        ("src.meta.arima.meta_arima", _ma),
        ("src.meta.arima._base", _base),
        ("src.meta.arima.multilabel_pca", _mlpca),
    ]:
        if old not in sys.modules:
            stubs[old] = new_mod

    for old, mod in stubs.items():
        sys.modules[old] = mod  # type: ignore[assignment]
    try:
        return joblib.load(buf)
    finally:
        for old in stubs:
            sys.modules.pop(old, None)
