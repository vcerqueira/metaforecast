from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from utilsforecast.evaluation import evaluate as uf_evaluate

from metaforecast.evaluation.aspects.rope import RopeAnalysis


class BaseModelRadar:
    """Shared CV-frame bookkeeping for ModelRadar.

    Parameters
    ----------
    cv_df : pd.DataFrame
        Cross-validation frame with ``unique_id``, ``ds``, ``y``, and one
        column per model (Nixtla convention).
    metrics : list of callable
        ``utilsforecast`` metric functions.  Multiple metrics are averaged.
    model_names : list of str or None
        Model columns to evaluate.  ``None`` auto-detects from ``cv_df``.
    agg_func : str, default 'mean'
        Aggregation for error scores (``'mean'`` or ``'median'``).
    id_col, time_col, target_col : str
        Standard Nixtla column names.
    """

    COLUMNS = {"metric": "metric", "horizon": "horizon", "cutoff": "cutoff"}
    DF_RESULT_COLUMNS = ["Model", "Result"]

    def __init__(
        self,
        cv_df: pd.DataFrame,
        metrics: list[Callable],
        model_names: list[str] | None = None,
        agg_func: str = "mean",
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ):
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col
        self.agg_func = agg_func
        self.metrics = metrics
        self.meta_data_cols: list[str] = []
        self.model_order: list[str] = []
        self.train_df = None

        self.cv_df = self._reset_on_uid(cv_df)
        self.cv_df = self._set_horizon_on_df(self.cv_df)
        self.models = self._get_model_names(cv_df) if model_names is None else model_names
        self.added_metadata = [
            col for col in cv_df.columns if col not in self.models + self.meta_data_cols
        ]

    def _set_horizon_on_df(self, cv: pd.DataFrame) -> pd.DataFrame:
        cv_ = cv.copy()
        co = self.COLUMNS["cutoff"]
        groups = [self.id_col, co] if co in cv_.columns else [self.id_col]
        dt_cols = [self.time_col, co] if co in cv_.columns else [self.time_col]
        for col in dt_cols:
            if not pd.api.types.is_datetime64_any_dtype(cv_[col]):
                cv_[col] = pd.to_datetime(cv_[col])
        cv_ = cv_.sort_values([*groups, self.time_col])
        cv_[self.COLUMNS["horizon"]] = cv_.groupby(groups).cumcount() + 1
        return cv_

    def _get_model_names(self, cv: pd.DataFrame) -> list[str]:
        self._set_meta_data()
        meta_cols_j = cv.columns.str.contains("|".join(self.meta_data_cols))
        candidates = cv.loc[:, ~meta_cols_j].select_dtypes(include=["number"])
        return [c for c in candidates.columns if c != "is_anomaly"]

    def _set_meta_data(self) -> None:
        self.meta_data_cols = [
            self.id_col,
            self.time_col,
            self.COLUMNS["cutoff"],
            self.COLUMNS["horizon"],
            self.target_col,
            "lo",
            "hi",
        ]

    def _reset_on_uid(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.index.name == self.id_col:
            return df.reset_index()
        return df

    @classmethod
    def _to_df_and_rename(cls, s: pd.Series) -> pd.DataFrame:
        df = s.reset_index()
        df.columns = cls.DF_RESULT_COLUMNS
        return df


class ModelRadarAcrossId:
    """Hard-series and tail-risk analysis on a per-UID error matrix.

    Parameters
    ----------
    reference : str or None
        Model used to define hard series (must be a column of ``err_df``).
    hardness_quantile : float, default 0.95
        Series whose reference error exceeds this quantile are "hard".
    cvar_quantile : float, default 0.9
        Quantile for expected shortfall (CVaR).
    agg_func : str, default 'mean'
        Aggregation used on hard series and in the tail.
    """

    def __init__(
        self,
        reference: str | None,
        hardness_quantile: float = 0.95,
        cvar_quantile: float = 0.9,
        agg_func: str = "mean",
    ):
        self.reference = reference
        self.hardness_quantile = hardness_quantile
        self.cvar_quantile = cvar_quantile
        self.hardness_threshold: float | None = None
        self.hard_uid: list = []
        self.agg_func = agg_func

    def get_hard_uids(self, err_df: pd.DataFrame, return_df: bool = True):
        """Identify series that are hard for the reference model."""
        if self.reference not in err_df.columns:
            raise ValueError(f"{self.reference} not in error columns")

        self.hardness_threshold = float(err_df[self.reference].quantile(self.hardness_quantile))
        hard = err_df.loc[err_df[self.reference] > self.hardness_threshold, :]
        self.hard_uid = hard.index.tolist()
        if return_df:
            return err_df.loc[self.hard_uid, :]
        return None

    def accuracy_on_hard(self, err_df: pd.DataFrame) -> pd.Series:
        """Mean (or median) error of each model on hard unique IDs."""
        err_df_uid = self.get_hard_uids(err_df=err_df)
        err_uid_avg = err_df_uid.agg(self.agg_func, numeric_only=True)
        err_uid_avg.name = "On Hard"
        return err_uid_avg

    def expected_shortfall(self, err_df: pd.DataFrame) -> pd.Series:
        """Expected shortfall (CVaR) of each model's per-UID errors."""
        shortfall = err_df.apply(lambda x: x[x > x.quantile(self.cvar_quantile)].agg(self.agg_func))
        shortfall.name = "Exp. Shortfall"
        return shortfall


class ModelRadar(BaseModelRadar):
    """Aspect-based evaluation of forecasting models on a CV frame.

    Slices error by overall mean, unique ID, forecast horizon, anomalies,
    and arbitrary groups, and attaches ROPE comparison plus hard-UID
    / CVaR analysis.

    Parameters
    ----------
    cv_df : pd.DataFrame
        Cross-validation forecasts in Nixtla format.
    metrics : list of callable
        ``utilsforecast`` metrics (e.g. ``smape``).
    model_names : list of str or None
        Model columns.  ``None`` auto-detects.
    hardness_reference : str or None
        Model used to flag hard series.
    ratios_reference : str or None
        Reference model for ROPE win/draw/loss ratios.
    rope : float, default 1.0
        ROPE width in percent.
    cvar_quantile : float, default 0.95
        Tail quantile for expected shortfall.
    hardness_quantile : float, default 0.9
        Quantile for hard unique IDs.
    train_df : pd.DataFrame, optional
        Training data forwarded to ``utilsforecast.evaluation.evaluate``
        (needed for scaled metrics such as MASE).
    agg_func : str, default 'mean'
        Score aggregation.
    id_col, time_col, target_col : str
        Nixtla column names.

    References
    ----------
    Cerqueira, V., Roque, L., & Soares, C. (2025). "Modelradar: aspect-based
    forecast evaluation." Machine Learning, 114(10), 229.

    Examples
    --------
    >>> from utilsforecast.losses import smape
    >>> from metaforecast.evaluation.aspects import ModelRadar
    >>>
    >>> radar = ModelRadar(
    ...     cv_df=cv,
    ...     metrics=[smape],
    ...     hardness_reference="SeasonalNaive",
    ...     ratios_reference="NHITS",
    ...     rope=10,
    ... )
    >>> overall = radar.evaluate()
    >>> err = radar.evaluate(keep_uids=True)
    >>> radar.rope.get_winning_ratios(err)
    """

    def __init__(
        self,
        cv_df: pd.DataFrame,
        metrics: list[Callable],
        model_names: list[str] | None = None,
        hardness_reference: str | None = None,
        ratios_reference: str | None = None,
        rope: float = 1.0,
        cvar_quantile: float = 0.95,
        hardness_quantile: float = 0.9,
        train_df: pd.DataFrame | None = None,
        agg_func: str = "mean",
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ):
        super().__init__(
            cv_df=cv_df,
            metrics=metrics,
            model_names=model_names,
            agg_func=agg_func,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
        )
        self.uid_accuracy = ModelRadarAcrossId(
            reference=hardness_reference,
            cvar_quantile=cvar_quantile,
            hardness_quantile=hardness_quantile,
            agg_func=agg_func,
        )
        self.rope = RopeAnalysis(reference=ratios_reference, rope=rope)
        self.train_df = train_df
        self.model_order = self.evaluate().sort_values().index.tolist()

    def evaluate(
        self,
        cv: pd.DataFrame | None = None,
        keep_uids: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """Score models with the configured metrics.

        Parameters
        ----------
        cv : pd.DataFrame, optional
            Subset of the CV frame.  Defaults to the full ``cv_df``.
        keep_uids : bool, default False
            If True, return one row per unique ID; otherwise a Series of
            overall scores named ``"Overall"``.
        """
        cv_ = self.cv_df if cv is None else cv
        scores_df = uf_evaluate(
            df=cv_, models=self.models, metrics=self.metrics, train_df=self.train_df
        )
        if keep_uids:
            return scores_df.groupby(self.id_col).agg(self.agg_func, numeric_only=True)

        scores_df = scores_df.drop(columns=[self.id_col, self.COLUMNS["metric"]]).agg(
            self.agg_func, numeric_only=True
        )
        scores_df.name = "Overall"
        return scores_df

    def evaluate_by_horizon(
        self,
        cv: pd.DataFrame | None = None,
        group_by_freq: bool = False,
        freq_col: str = "Frequency",
    ) -> pd.DataFrame:
        """Score models at each forecast horizon (cumulative up to *h*)."""
        cv_ = self.cv_df if cv is None else cv
        cv_groups = cv_.groupby(freq_col) if group_by_freq else [(None, cv_)]

        scores_by_group = {}
        h_col = self.COLUMNS["horizon"]
        for g, df in cv_groups:
            fh = df[h_col].sort_values().unique()
            eval_horizon = {h: self.evaluate(df.loc[df[h_col] <= h]) for h in fh}
            scores_by_group[g] = pd.DataFrame(eval_horizon).T

        scores_df = pd.concat(scores_by_group)
        if group_by_freq:
            return scores_df.reset_index()

        scores_df = scores_df.reset_index(drop=True)
        scores_df[h_col] = np.arange(1, scores_df.shape[0] + 1)
        return scores_df

    def evaluate_by_horizon_bounds(self, cv: pd.DataFrame | None = None) -> pd.DataFrame:
        """Compare first-step vs. last-step error for each unique ID."""
        cv_ = self.cv_df if cv is None else cv
        sort_cols = [self.id_col, self.COLUMNS["horizon"]]
        if self.COLUMNS["cutoff"] in cv_:
            sort_cols = [self.id_col, self.COLUMNS["cutoff"], self.COLUMNS["horizon"]]
        sorted_cv = cv_.sort_values(sort_cols)

        first_horizon = sorted_cv.groupby(self.id_col).first().reset_index()
        last_horizon = sorted_cv.groupby(self.id_col).last().reset_index()

        errors_first_df = self._to_df_and_rename(self.evaluate(first_horizon)).rename(
            columns={"Result": "First horizon"}
        )
        errors_last_df = self._to_df_and_rename(self.evaluate(last_horizon)).rename(
            columns={"Result": "Last horizon"}
        )
        return errors_first_df.merge(errors_last_df, on=self.DF_RESULT_COLUMNS[0]).set_index(
            "Model"
        )

    def evaluate_by_anomaly(
        self,
        cv: pd.DataFrame | None = None,
        mode: str = "observations",
        anomaly_col: str = "is_anomaly",
    ) -> pd.DataFrame | None:
        """Score models on anomalous observations or on series that contain them.

        Parameters
        ----------
        mode : {'observations', 'series'}
            ``observations`` keeps only anomalous rows; ``series`` keeps the
            full series if it has at least one anomaly.
        anomaly_col : str, default 'is_anomaly'
            Numeric 0/1 column on the CV frame.
        """
        cv_ = self.cv_df if cv is None else cv
        if mode not in {"observations", "series"}:
            raise ValueError("mode must be either 'observations' or 'series'")

        scores_uids = {}
        for uid, df_uid in cv_.groupby(self.id_col):
            if df_uid[anomaly_col].sum() <= 0:
                continue
            uid_ = df_uid.loc[df_uid[anomaly_col] > 0] if mode == "observations" else df_uid
            if not uid_.empty:
                scores_uids[uid] = self.evaluate(uid_)

        if not scores_uids:
            return None
        return pd.DataFrame(scores_uids).T

    def evaluate_by_group(
        self,
        group_col: str,
        cv: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Score models within each level of a categorical column."""
        cv_ = self.cv_df if cv is None else cv
        if group_col not in cv_.columns:
            raise KeyError(f"Column '{group_col}' not found in DataFrame")

        results_by_group = {
            group: self.evaluate(group_df) for group, group_df in cv_.groupby(group_col)
        }
        return pd.concat(results_by_group, axis=1)
