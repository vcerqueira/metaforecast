"""Internal base classes and utilities for MetaARIMA.

Contains:

* :func:`tsfeatures_uid` -- single-series feature extraction.
* :class:`MetaARIMAUtils` -- ARIMA model enumeration and diagnostics.
* :class:`_MetaARIMABase` -- exhaustive AICc selector.
* :class:`_HalvingMetaARIMABase` -- successive-halving selector.
"""

from __future__ import annotations

import warnings
from math import log
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsforecast import StatsForecast
from statsforecast.models import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from tsfeatures import (
    acf_features,
    arch_stat,
    crossing_points,
    entropy,
    flat_spots,
    heterogeneity,
    holt_parameters,
    hurst,
    hw_parameters,
    lumpiness,
    nonlinearity,
    pacf_features,
    scalets,
    series_length,
    stability,
    stl_features,
    unitroot_kpss,
    unitroot_pp,
)

warnings.filterwarnings(action="ignore")

BEST_CATBOOST_PARAMS = {
    'monthly': {'bootstrap_type': 'Bernoulli',
                'border_count': 32,
                'depth': 4,
                'eval_metric': 'MultiRMSE',
                'iterations': 283,
                'l2_leaf_reg': 16.732679992180078,
                'leaf_estimation_iterations': 2,
                'learning_rate': 0.04248616344796388,
                'loss_function': 'MultiRMSE',
                'model_size_reg': 2.0916597401644705,
                'od_type': 'Iter',
                'od_wait': 60,
                'random_seed': 42,
                'rsm': 0.6406208143252348,
                'task_type': 'CPU',
                'use_best_model': False,
                'verbose': False},
    'quarterly': {'bootstrap_type': 'Bernoulli',
                  'border_count': 32,
                  'depth': 4,
                  'eval_metric': 'MultiRMSE',
                  'iterations': 266,
                  'l2_leaf_reg': 6.032217388079633,
                  'leaf_estimation_iterations': 2,
                  'learning_rate': 0.05961229266653637,
                  'loss_function': 'MultiRMSE',
                  'model_size_reg': 0.6153682157721689,
                  'od_type': 'Iter',
                  'od_wait': 20,
                  'random_seed': 42,
                  'rsm': 0.6992272121986959,
                  'task_type': 'CPU',
                  'use_best_model': False,
                  'verbose': False},
    'yearly': {'bootstrap_type': 'Bernoulli',
               'border_count': 64,
               'depth': 4,
               'eval_metric': 'MultiRMSE',
               'iterations': 300,
               'l2_leaf_reg': 5.903373826675397,
               'leaf_estimation_iterations': 2,
               'learning_rate': 0.05544986984539306,
               'loss_function': 'MultiRMSE',
               'model_size_reg': 1.5546708502742612,
               'od_type': 'Iter',
               'od_wait': 70,
               'random_seed': 42,
               'rsm': 0.8381786985397336,
               'task_type': 'CPU',
               'use_best_model': False,
               'verbose': False}
}

FEATURE_ORDER = [
    "hurst",
    "series_length",
    "unitroot_pp",
    "unitroot_kpss",
    "hw_alpha",
    "hw_beta",
    "hw_gamma",
    "stability",
    "nperiods",
    "seasonal_period",
    "trend",
    "spike",
    "linearity",
    "curvature",
    "e_acf1",
    "e_acf10",
    "seasonal_strength",
    "peak",
    "trough",
    "x_pacf5",
    "diff1x_pacf5",
    "diff2x_pacf5",
    "seas_pacf",
    "nonlinearity",
    "lumpiness",
    "alpha",
    "beta",
    "arch_acf",
    "garch_acf",
    "arch_r2",
    "garch_r2",
    "flat_spots",
    "entropy",
    "crossing_points",
    "arch_lm",
    "x_acf1",
    "x_acf10",
    "diff1_acf1",
    "diff1_acf10",
    "diff2_acf1",
    "diff2_acf10",
    "seas_acf1",
]

SEAS_FEATURES = ["seasonal_strength", "peak", "trough", "seas_pacf", "seas_acf1"]
FEATURE_ORDER_NON_SEAS = [x for x in FEATURE_ORDER if x not in SEAS_FEATURES]

ORDER_MAX = {"AR": 4, "I": 1, "MA": 4, "S_AR": 1, "S_I": 1, "S_MA": 1}
ORDER_MAX_NONSEASONAL = {"AR": 4, "I": 1, "MA": 4, "S_AR": 0, "S_I": 0, "S_MA": 0}


def tsfeatures_uid(
        uid_df: pd.DataFrame,
        freq: int,
        impute_seas: bool = False,
        target_col: str = "y",
        id_col: str = "unique_id",
) -> pd.DataFrame:
    """Extract tsfeatures for a single time series.

    Parameters
    ----------
    uid_df : pd.DataFrame
        DataFrame for a single ``unique_id`` with columns ``[unique_id, ds, y]``.
    freq : int
        Seasonal period (e.g. 12 for monthly, 4 for quarterly).
    impute_seas : bool, default False
        If True, add seasonal features as NaN when the series is non-seasonal.
    target_col, id_col : str
        Column names.

    Returns
    -------
    pd.DataFrame
        Single-row feature DataFrame indexed by ``unique_id``.
    """
    x_r = uid_df[target_col].values
    x = scalets(x_r)

    feats = {
        **hurst(x, freq),
        **series_length(x, freq),
        **unitroot_pp(x, freq),
        **unitroot_kpss(x, freq),
        **hw_parameters(x, freq),
        **stability(x, freq),
        **stl_features(x, freq),
        **pacf_features(x, freq),
        **nonlinearity(x, freq),
        **lumpiness(x, freq),
        **holt_parameters(x, freq),
        **heterogeneity(x, freq),
        **flat_spots(x, freq),
        **entropy(x, freq),
        **crossing_points(x, freq),
        **arch_stat(x, freq),
        **acf_features(x, freq),
    }

    feats_series = pd.Series(feats)

    if impute_seas:
        for f in FEATURE_ORDER:
            if f not in feats_series.index:
                feats_series[f] = np.nan
        feats_series = feats_series[FEATURE_ORDER]
    elif freq > 1:
        feats_series = feats_series[FEATURE_ORDER]
    else:
        feats_series = feats_series[FEATURE_ORDER_NON_SEAS]

    feats_df = pd.DataFrame(feats_series).T.fillna(-1)
    feats_df.index = [uid_df[id_col].values[0]]

    return feats_df


class MetaARIMAUtils:
    """Utility methods for ARIMA configuration enumeration and diagnostics."""

    @staticmethod
    def get_model_order(mod, as_alias: bool = False, alias_freq: int = 1) -> pd.Series | str:
        """Extract the (p,d,q)(P,D,Q)[m] order from a fitted statsforecast ARIMA.

        Parameters
        ----------
        mod : dict
            The ``model_`` dict from a fitted ``statsforecast.models.ARIMA``.
        as_alias : bool
            If True, return a human-readable alias string.
        alias_freq : int
            Seasonal period to embed in the alias.
        """
        order = tuple(mod["arma"][i] for i in [0, 5, 1, 2, 6, 3, 4])
        o = pd.Series(order, index=["AR", "I", "MA", "S_AR", "S_I", "S_MA", "m"])

        if as_alias:
            return (
                f"ARIMA({o['AR']},{o['I']},{o['MA']})"
                f"({o['S_AR']},{o['S_I']},{o['S_MA']})[{alias_freq}]"
            )
        return o

    @classmethod
    def get_models_sf(
            cls,
            season_length: int,
            return_names: bool = False,
            max_config: Dict | None = None,
            alias_list: List[str] | None = None,
    ) -> list:
        """Enumerate all ARIMA configurations up to ``max_config``.

        Parameters
        ----------
        season_length : int
            Seasonal period for the ARIMA models.
        return_names : bool
            If True, return alias strings instead of model objects.
        max_config : dict, optional
            Upper bounds for each order component.  Defaults to
            ``ORDER_MAX``.
        alias_list : list of str, optional
            If given, filter to only these aliases.

        Returns
        -------
        list
            List of ``statsforecast.models.ARIMA`` instances (or alias
            strings if ``return_names=True``).
        """
        max_config_ = ORDER_MAX if max_config is None else max_config

        models = []
        for ar in range(max_config_["AR"] + 1):
            for i in range(max_config_["I"] + 1):
                for ma in range(max_config_["MA"] + 1):
                    for s_ar in range(max_config_["S_AR"] + 1):
                        for s_i in range(max_config_["S_I"] + 1):
                            for s_ma in range(max_config_["S_MA"] + 1):
                                alias = (
                                    f"ARIMA({ar},{i},{ma})({s_ar},{s_i},{s_ma})[{season_length}]"
                                )
                                models.append(
                                    ARIMA(
                                        order=(ar, i, ma),
                                        season_length=season_length,
                                        seasonal_order=(s_ar, s_i, s_ma),
                                        alias=alias,
                                    )
                                )

        if return_names:
            return [x.alias for x in models]

        if alias_list is not None:
            models = [x for x in models if x.alias in alias_list]

        return models

    @classmethod
    def model_summary(cls, model) -> dict:
        """Extract coefficients, goodness-of-fit, and residual diagnostics."""
        coefs = {f"coef_{k}": model["coef"][k] for k in model["coef"]}

        try:
            var_coef_avg = model["var_coef"].mean()
        except AttributeError:
            var_coef_avg = np.nan

        goodness_fit = {
            "var_coef_mean": var_coef_avg,
            "aic": model["aic"],
            "aicc": model["aicc"],
            "bic": model["bic"],
            "loglik": model["loglik"],
        }

        resid_tests = cls.residual_testing(model["residuals"])

        return {**coefs, **goodness_fit, **resid_tests}

    @staticmethod
    def residual_testing(residuals) -> dict:
        """Diagnostic p-values for model residuals.

        Returns
        -------
        dict
            Keys: ``zero_mean``, ``constant_variance``, ``normality``,
            ``no_autocorrelation``.  Values > 0.05 indicate acceptable
            residuals.
        """
        mean_test = stats.ttest_1samp(residuals, 0).pvalue

        squared_resid = residuals ** 2
        trend = np.arange(len(residuals))
        bp_test = stats.linregress(trend, squared_resid).pvalue

        jb_test = jarque_bera(residuals)[1]

        lb_test = acorr_ljungbox(residuals, lags=[1]).lb_pvalue.values[0]

        return {
            "zero_mean": mean_test,
            "constant_variance": bp_test,
            "normality": jb_test,
            "no_autocorrelation": lb_test,
        }


class _MetaARIMABase:
    """Exhaustive AICc-based ARIMA selector.

    Fits every configuration in ``config_space``, selects the one with
    the lowest AICc.
    """

    def __init__(self, config_space: List[str], season_length: int, freq: str):
        self.config_space = config_space
        self.freq = freq
        self.season_length = season_length
        self.models = MetaARIMAUtils.get_models_sf(
            season_length=self.season_length, alias_list=self.config_space
        )
        self.nmodels = len(self.models)
        self.sf = StatsForecast(models=self.models, freq=self.freq)
        self.alias = "MetaARIMA"

    def fit(self, df: pd.DataFrame):
        self.sf.fit(df=df)
        aicc_ = [self.sf.fitted_[0][i].model_["aicc"] for i in range(self.nmodels)]
        best_idx = np.array(aicc_).argmin()
        self.sf.fitted_ = np.array([[self.sf.fitted_[0][best_idx]]])
        self.sf.fitted_[0][0].alias = self.alias

    def predict(self, h: int, level: List | None = None):
        if level is not None:
            return self.sf.predict(h=h, level=level)
        return self.sf.predict(h=h)


class _HalvingMetaARIMABase(_MetaARIMABase):
    """Successive-halving ARIMA selector.

    Evaluates configurations on increasing subsets of the data,
    eliminating the worst fraction at each round.

    Parameters
    ----------
    config_space : list of str
        ARIMA alias strings.
    season_length : int
        Seasonal period.
    freq : str
        Time series frequency string.
    eta : float, default 2
        Proportion of configurations eliminated per round.
    resource_factor : float, default 2
        Multiplier for sample size increase each round.
    init_resource_factor : int, default 4
        Initial sample size = ``season_length * init_resource_factor``.
    min_configs : int, default 1
        Minimum configurations to keep.
    eval_mstl : bool, default False
        Reserved for future use.
    """

    def __init__(
            self,
            config_space: List[str],
            season_length: int,
            freq: str,
            eta: float = 2,
            resource_factor: float = 2,
            init_resource_factor: int = 4,
            min_configs: int = 1,
            eval_mstl: bool = False,
    ):
        super().__init__(config_space, season_length, freq)
        self.eta = eta
        self.resource_factor = resource_factor
        self.min_configs = min_configs
        self.init_resource_factor = init_resource_factor
        self.eval_mstl = eval_mstl
        self.tot_nobs = 0

    def _evaluate_models(
            self, df: pd.DataFrame, model_indices: List[int], sample_size: int
    ) -> List[Tuple[int, float]]:
        df_subset = df.tail(sample_size).copy()
        models_subset = [self.models[i] for i in model_indices]
        sf_subset = StatsForecast(models=models_subset, freq=self.freq)
        sf_subset.fit(df=df_subset)

        results = []
        for i, model_idx in enumerate(model_indices):
            aicc = sf_subset.fitted_[0][i].model_["aicc"]
            results.append((model_idx, aicc))

        return results

    def fit(self, df: pd.DataFrame):
        """Fit using successive halving to select the best ARIMA config."""
        assert self.nmodels > 0, "nmodels is not positive"

        n_rows = df.shape[0]
        n_models = self.nmodels
        s_max = int(log(n_models, self.eta)) if n_models > 1 else 0

        remaining_indices = list(range(n_models))

        min_sample_size = (
            self.season_length * self.init_resource_factor if self.season_length > 1 else 20
        )

        self.tot_nobs = 0
        for s in range(s_max + 1):
            sample_size = min(n_rows, int(min_sample_size * (self.resource_factor ** s)))

            eval_results = self._evaluate_models(df, remaining_indices, int(sample_size))
            self.tot_nobs += len(remaining_indices) * int(sample_size)

            eval_results.sort(key=lambda x: x[1])
            n_keep = max(int(n_models / (self.eta ** (s + 1))), self.min_configs)
            remaining_indices = [idx for idx, _ in eval_results[:n_keep]]

            if len(remaining_indices) <= 1:
                break

        best_idx = remaining_indices[0]

        self.sf = StatsForecast(models=[self.models[best_idx]], freq=self.freq)
        self.sf.fit(df=df)
        self.tot_nobs += df.shape[0]
        self.sf.fitted_[0][0].alias = self.alias
