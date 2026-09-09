"""Graph-based semi-synthetic time series generation (Grasynda).

Quantile-transition graphs learned from STL components are sampled to
produce new series that preserve local dynamics of the source data.
"""

from __future__ import annotations

import copy
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from statsmodels.tsa.seasonal import STL

from metaforecast.synth.generators.base import SemiSyntheticGenerator

_VALID_COMPONENTS = ("remainder", "trend", "seasonal", "y")
_VALID_SAMPLING = ("discrete", "kde")


class Grasynda(SemiSyntheticGenerator):
    """Generate synthetic time series from quantile-transition graphs.

    Each source series is decomposed with STL (unless the raw target is
    modelled directly).  Values of selected components are binned into
    quantiles, a Markov transition matrix is estimated, and a new series
    is obtained by walking the chain and sampling within each bin.

    References
    ----------
    Amorim, L., Santos, M., Azevedo, P. J., Soares, C., & Cerqueira, V.
    (2026). "Grasynda: Graph-Based Synthetic Time Series Generation."
    International Symposium on Intelligent Data Analysis.

    Examples
    --------
    >>> import pandas as pd
    >>> from datasetsforecast.m3 import M3
    >>> from neuralforecast import NeuralForecast
    >>> from neuralforecast.models import NHITS
    >>>
    >>> from metaforecast.synth import Grasynda
    >>> from metaforecast.evaluation.cv import SeriesWiseSplit
    >>>
    >>> df, *_ = M3.load('.', group='Monthly')
    >>>
    >>> horizon = 12
    >>> train, test = SeriesWiseSplit.time_wise_split(df, horizon)
    >>>
    >>> tsgen = Grasynda(period=12)
    >>> synth_df = tsgen.transform(train)
    >>> train_aug = pd.concat([train, synth_df])
    >>>
    >>> models = [NHITS(input_size=horizon, h=horizon, accelerator='cpu')]
    >>> nf = NeuralForecast(models=models, freq='M')
    >>> nf.fit(df=train_aug)
    >>> fcst = nf.predict(df=train)
    """

    def __init__(
        self,
        period: int,
        n_quantiles: int = 25,
        components_to_model: list[str] | None = None,
        component_params: dict[str, dict] | None = None,
        sampling_type: str = "kde",
        ensemble_transitions: bool = False,
        ensemble_size: int = 5,
        robust: bool = False,
    ):
        """Initialize a GraSyNDA generator.

        Parameters
        ----------
        period : int
            Seasonal period passed to STL (e.g. 12 for monthly data).
        n_quantiles : int, default 25
            Number of quantile bins per series.
        components_to_model : list of str, optional
            STL components to synthesise.  Allowed values: ``remainder``,
            ``trend``, ``seasonal``, ``y``.  Defaults to ``["remainder"]``.
            Use ``["y"]`` to skip STL and model the raw series.
        component_params : dict, optional
            Per-component overrides of ``n_quantiles``, ``sampling_type``,
            ``ensemble_transitions``, or ``ensemble_size``.
        sampling_type : {'discrete', 'kde'}, default 'discrete'
            How values are drawn inside a quantile bin.
        ensemble_transitions : bool, default False
            Average each series' transition matrix with its nearest
            neighbours (by Frobenius distance).
        ensemble_size : int, default 5
            Neighbourhood size when ``ensemble_transitions`` is True.
        robust : bool, default False
            Use robust STL.
        """
        super().__init__(alias="Grasynda")

        if sampling_type not in _VALID_SAMPLING:
            raise ValueError(f"sampling_type must be one of {_VALID_SAMPLING}")

        self.period = period
        self.n_quantiles = n_quantiles
        self.components_to_model = (
            list(components_to_model) if components_to_model else ["remainder"]
        )
        unknown = set(self.components_to_model) - set(_VALID_COMPONENTS)
        if unknown:
            raise ValueError(f"Unknown components: {unknown}. Allowed: {_VALID_COMPONENTS}")
        if "y" in self.components_to_model and len(self.components_to_model) > 1:
            raise ValueError("Cannot mix 'y' with STL components.")

        self.component_params = component_params or {}
        self.global_settings = {
            "n_quantiles": n_quantiles,
            "sampling_type": sampling_type,
            "ensemble_transitions": ensemble_transitions,
            "ensemble_size": ensemble_size,
            "robust": robust,
        }

        self.transition_mats: dict[str, dict] = {}
        self.ensemble_transition_mats: dict[str, dict] = {}
        self.uid_pw_distance: dict[tuple, float] = {}

    def _get_param(self, component: str, param: str):
        """Return a per-component parameter, falling back to global settings."""
        if component in self.component_params and param in self.component_params[component]:
            return self.component_params[component][param]
        return self.global_settings.get(param)

    def transform(self, df: pd.DataFrame, n_series: int = -1, **kwargs) -> pd.DataFrame:
        """Generate synthetic series from quantile-transition graphs.

        Parameters
        ----------
        df : pd.DataFrame
            Source dataset with ``unique_id``, ``ds``, ``y`` columns.
        n_series : int, default -1
            Number of synthetic series to generate.  ``-1`` produces one
            series per input unique ID.

        Returns
        -------
        pd.DataFrame
            Synthetic dataset in nixtla format.
        """
        self._assert_datatypes(df)

        skip_decomposition = "y" in self.components_to_model
        df_ = df.copy()
        if not skip_decomposition:
            df_ = self._decompose(
                df_, period=self.period, robust=bool(self.global_settings["robust"])
            )

        uids = df_[self.id_col].unique()
        if n_series < 0:
            selected = list(uids)
        else:
            replace = n_series > len(uids)
            selected = np.random.choice(uids, size=n_series, replace=replace).tolist()

        self.transition_mats = {}
        self.ensemble_transition_mats = {}
        self.uid_pw_distance = {}

        prepared = {}
        for component in self.components_to_model:
            if component not in df_.columns:
                raise ValueError(f"Component '{component}' not found in DataFrame.")

            comp_df = df_.copy()
            comp_df["Quantile"] = self._get_quantiles(comp_df, component, component=component)
            self._calc_transition_matrix(comp_df, component=component)

            if self._get_param(component, "ensemble_transitions"):
                self.ensemble_transition_mats[component] = self._get_ensemble_transition_mats(
                    component=component,
                    ensemble_size=self._get_param(component, "ensemble_size"),
                )
            prepared[component] = comp_df

        synth_rows = []
        for uid in selected:
            synth_components = {}
            for component, comp_df in prepared.items():
                synth_components[component] = self._create_synthetic_ts_quantile(
                    comp_df,
                    uid,
                    component,
                    component,
                    sampling_type=self._get_param(component, "sampling_type"),
                )
            synth_rows.append(self._reconstruct_uid(df_, uid, synth_components, skip_decomposition))

        return pd.concat(synth_rows).reset_index(drop=True)

    def _create_synthetic_ts(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate synthetic series from a (possibly multi-uid) dataset."""
        return self.transform(df, n_series=kwargs.get("n_series", -1))

    def _decompose(self, df: pd.DataFrame, period: int, robust: bool = False) -> pd.DataFrame:
        """STL-decompose each series into seasonal, trend, and remainder."""
        parts = []
        for uid, group in df.groupby(self.id_col):
            result = STL(group[self.target_col], period=period, robust=robust).fit()
            parts.append(
                pd.DataFrame(
                    {
                        self.id_col: uid,
                        self.time_col: group[self.time_col].values,
                        "seasonal": result.seasonal,
                        "trend": result.trend,
                        "remainder": result.resid,
                        self.target_col: group[self.target_col].values,
                    }
                )
            )
        return pd.concat(parts, ignore_index=True)

    def _reconstruct_uid(
        self,
        df: pd.DataFrame,
        uid,
        synth_components: dict,
        skip_decomposition: bool,
    ) -> pd.DataFrame:
        """Rebuild one series, replacing modelled components with synthetic ones."""
        row = df[df[self.id_col] == uid].copy()
        for comp, series in synth_components.items():
            row[comp] = series.values

        if not skip_decomposition:
            row[self.target_col] = row["trend"] + row["seasonal"] + row["remainder"]

        cols = [self.time_col, self.id_col, self.target_col]
        row = row[cols].copy()
        row[self.id_col] = f"{uid}_{self.alias}{self.counter}"
        self.counter += 1
        return row

    def _get_quantiles(self, df: pd.DataFrame, target_col: str, component: str) -> pd.Series:
        n_q = self._get_param(component, "n_quantiles")
        return (
            df.groupby(self.id_col)[target_col]
            .transform(lambda x: pd.qcut(x, n_q, labels=False, duplicates="drop"))
            .fillna(0)
            .astype(int)
        )

    def _calc_transition_matrix(self, df: pd.DataFrame, component: str) -> None:
        self.transition_mats[component] = {}
        n_q = self._get_param(component, "n_quantiles")
        for uid, group in df.groupby(self.id_col):
            quantiles = group["Quantile"].to_numpy()
            counts = np.zeros((n_q, n_q))
            for i in range(len(quantiles) - 1):
                counts[quantiles[i], quantiles[i + 1]] += 1

            with np.errstate(divide="ignore", invalid="ignore"):
                probs = counts / counts.sum(axis=1, keepdims=True)
            probs = np.nan_to_num(probs)
            empty = probs.sum(axis=1) == 0
            probs[empty] = 1.0 / n_q
            self.transition_mats[component][uid] = probs

    def _get_ensemble_transition_mats(self, component: str, ensemble_size: int) -> dict:
        mats = copy.deepcopy(self.transition_mats[component])
        for uid in mats:
            self.uid_pw_distance[(uid, uid)] = 0.0
        for uid1, uid2 in combinations(mats, 2):
            dist = float(np.linalg.norm(mats[uid1] - mats[uid2]))
            self.uid_pw_distance[(uid1, uid2)] = dist
            self.uid_pw_distance[(uid2, uid1)] = dist

        ensemble_mats = {}
        for uid in mats:
            uid_dists = pd.Series({other: self.uid_pw_distance[(uid, other)] for other in mats})
            similar = uid_dists.sort_values().head(ensemble_size).index.tolist()
            ensemble_mats[uid] = np.sum([mats[u] for u in similar], axis=0) / len(similar)
        return ensemble_mats

    def matrix_to_edgelist(
        self,
        uid: str,
        component: str | None = None,
        threshold: float = 0,
    ) -> pd.DataFrame:
        """Return a quantile-transition matrix as an edge list.

        Parameters
        ----------
        uid : str
            Source series identifier (before aliasing).
        component : str, optional
            Component whose matrix to export.  Required when more than
            one component was modelled.
        threshold : float, default 0
            Drop edges with weight at or below this value.
        """
        if component is None:
            if len(self.transition_mats) != 1:
                raise ValueError("Specify component when multiple transition matrices exist.")
            component = next(iter(self.transition_mats))

        adj = self.transition_mats[component][uid]
        from_idx, to_idx = np.nonzero(adj > threshold)
        return pd.DataFrame({"from": from_idx, "to": to_idx, "weight": adj[from_idx, to_idx]})

    def _transition_matrix_for(self, component: str, uid) -> np.ndarray:
        if (
            component in self.ensemble_transition_mats
            and uid in self.ensemble_transition_mats[component]
        ):
            return self.ensemble_transition_mats[component][uid]
        return self.transition_mats[component][uid]

    def _generate_quantile_series(self, uid_df: pd.DataFrame, uid, component: str) -> np.ndarray:
        mat = self._transition_matrix_for(component, uid)
        n_q = self._get_param(component, "n_quantiles")
        q_series = np.zeros(len(uid_df), dtype=int)
        q_series[0] = uid_df["Quantile"].to_numpy()[0]
        for t in range(1, len(q_series)):
            probs = mat[q_series[t - 1]]
            probs = np.ones(n_q) / n_q if probs.sum() == 0 else probs / probs.sum()
            q_series[t] = np.random.choice(np.arange(n_q), p=probs)
        return q_series

    @staticmethod
    def _build_safe_kde(vals: np.ndarray):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size <= 1 or np.unique(vals).size <= 1:
            return None
        return gaussian_kde(vals)

    def _create_synthetic_ts_quantile(
        self,
        df: pd.DataFrame,
        uid,
        component: str,
        target_col: str,
        sampling_type: str,
    ) -> pd.Series:
        uid_df = df[df[self.id_col] == uid]
        q_path = self._generate_quantile_series(uid_df, uid, component)
        uid_vals = uid_df[target_col]
        uid_quantiles = uid_df["Quantile"]
        n_q = self._get_param(component, "n_quantiles")

        bin_props = {}
        for q in range(n_q):
            vals = uid_vals[uid_quantiles == q].to_numpy()
            if len(vals) > 0:
                bin_props[q] = {
                    "vals": vals,
                    "min": vals.min(),
                    "max": vals.max(),
                    "kde": self._build_safe_kde(vals) if sampling_type == "kde" else None,
                }
            else:
                bin_props[q] = None

        synth_vals = np.zeros(len(uid_vals))
        synth_vals[0] = uid_vals.to_numpy()[0]
        for i in range(1, len(uid_vals)):
            props = bin_props.get(int(q_path[i]))
            if props is None:
                synth_vals[i] = synth_vals[i - 1]
                continue
            if sampling_type == "discrete":
                synth_vals[i] = np.random.choice(props["vals"])
            elif sampling_type == "kde":
                if props["kde"] is not None:
                    synth_vals[i] = props["kde"].resample(1)[0][0]
                else:
                    synth_vals[i] = np.random.choice(props["vals"])
            else:
                raise ValueError(f"Unknown sampling_type: {sampling_type}")

        return pd.Series(synth_vals, index=uid_df.index)
