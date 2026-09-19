from __future__ import annotations

from collections.abc import Callable, Sequence

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
        ``utilsforecast`` metric functions.  Multiple metrics are averaged
        unless ``keep_metrics=True`` is passed to :meth:`ModelRadar.evaluate`.
    model_names : list of str or None
        Model columns to evaluate.  ``None`` auto-detects from ``cv_df``.
    agg_func : str, default 'mean'
        Aggregation for error scores (``'mean'`` or ``'median'``).
    id_col, time_col, target_col : str
        Standard Nixtla column names.
    """

    COLUMNS = {"metric": "metric", "horizon": "horizon", "cutoff": "cutoff"}
    DF_RESULT_COLUMNS = ["Model", "Result"]
    EXTRA_NON_MODEL_COLS = ("lo", "hi", "is_anomaly", "h", "index")

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
        self.train_df = None

        self._set_meta_data()
        self.cv_df = self._reset_on_uid(cv_df)
        self.cv_df = self._set_horizon_on_df(self.cv_df)
        self.models = (
            self._get_model_names(self.cv_df) if model_names is None else list(model_names)
        )
        model_or_meta = set(self.models) | set(self.meta_data_cols)
        self.added_metadata = [col for col in self.cv_df.columns if col not in model_or_meta]

    def _set_horizon_on_df(self, cv: pd.DataFrame) -> pd.DataFrame:
        cv_ = cv.copy()
        h_col = self.COLUMNS["horizon"]
        if h_col in cv_.columns:
            return cv_

        co = self.COLUMNS["cutoff"]
        groups = [self.id_col, co] if co in cv_.columns else [self.id_col]
        dt_cols = [self.time_col] + ([co] if co in cv_.columns else [])
        for col in dt_cols:
            if pd.api.types.is_numeric_dtype(cv_[col]):
                continue
            if not pd.api.types.is_datetime64_any_dtype(cv_[col]):
                cv_[col] = pd.to_datetime(cv_[col])
        cv_ = cv_.sort_values([*groups, self.time_col])
        cv_[h_col] = cv_.groupby(groups, sort=False).cumcount() + 1
        return cv_

    def _get_model_names(self, cv: pd.DataFrame) -> list[str]:
        exclude = set(self.meta_data_cols)
        candidates = cv.select_dtypes(include=["number"])
        return [c for c in candidates.columns if c not in exclude]

    def _set_meta_data(self) -> None:
        self.meta_data_cols = [
            self.id_col,
            self.time_col,
            self.COLUMNS["cutoff"],
            self.COLUMNS["horizon"],
            self.COLUMNS["metric"],
            self.target_col,
            *self.EXTRA_NON_MODEL_COLS,
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
    hardness_quantile : float, default 0.9
        Series whose reference error exceeds this quantile are "hard".
    cvar_quantile : float, default 0.95
        Quantile for expected shortfall (CVaR).
    agg_func : str, default 'mean'
        Aggregation used on hard series and in the tail.
    """

    def __init__(
        self,
        reference: str | None,
        hardness_quantile: float = 0.9,
        cvar_quantile: float = 0.95,
        agg_func: str = "mean",
    ):
        self.reference = reference
        self.hardness_quantile = hardness_quantile
        self.cvar_quantile = cvar_quantile
        self.hardness_threshold: float | None = None
        self.hard_uid: list = []
        self.agg_func = agg_func

    def get_hard_uids(self, err_df: pd.DataFrame, return_df: bool = True):
        """Identify series that are hard for the reference model.

        Parameters
        ----------
        err_df : pd.DataFrame
            Per-UID scores (index = unique IDs, columns = models).
        return_df : bool, default True
            If True, return the hard-series slice of ``err_df``.  If False,
            return the list of hard unique IDs.
        """
        if self.reference is None:
            raise ValueError("hardness reference model is not set")
        if self.reference not in err_df.columns:
            raise ValueError(f"{self.reference} not in error columns")

        self.hardness_threshold = float(err_df[self.reference].quantile(self.hardness_quantile))
        hard = err_df.loc[err_df[self.reference] > self.hardness_threshold, :]
        self.hard_uid = hard.index.tolist()
        if return_df:
            return err_df.loc[self.hard_uid, :]
        return self.hard_uid

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
    rope_reference : str or None
        Reference model for ROPE win/draw/loss ratios.  Takes precedence
        over ``ratios_reference``.
    ratios_reference : str or None
        Alias for ``rope_reference``.
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
    ...     rope_reference="NHITS",
    ...     rope=10,
    ... )
    >>> overall = radar.evaluate()
    >>> err = radar.evaluate_by_uid()
    >>> radar.winning_ratios()
    >>> radar.aspect_table()
    """

    def __init__(
        self,
        cv_df: pd.DataFrame,
        metrics: list[Callable],
        model_names: list[str] | None = None,
        hardness_reference: str | None = None,
        ratios_reference: str | None = None,
        rope_reference: str | None = None,
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
        rope_ref = self._resolve_rope_reference(rope_reference, ratios_reference)
        self._validate_reference(hardness_reference, "hardness_reference")
        self._validate_reference(rope_ref, "rope_reference")

        self.hardness = ModelRadarAcrossId(
            reference=hardness_reference,
            cvar_quantile=cvar_quantile,
            hardness_quantile=hardness_quantile,
            agg_func=agg_func,
        )
        self.uid_accuracy = self.hardness
        self.rope = RopeAnalysis(reference=rope_ref, rope=rope)
        self.train_df = None if train_df is None else self._reset_on_uid(train_df)
        self._uid_scores: pd.DataFrame | None = None
        self._model_order: list[str] | None = None

    @staticmethod
    def _resolve_rope_reference(
        rope_reference: str | None,
        ratios_reference: str | None,
    ) -> str | None:
        if (
            rope_reference is not None
            and ratios_reference is not None
            and rope_reference != ratios_reference
        ):
            raise ValueError("rope_reference and ratios_reference both set to different models")
        return rope_reference if rope_reference is not None else ratios_reference

    def _validate_reference(self, reference: str | None, name: str) -> None:
        if reference is not None and reference not in self.models:
            raise ValueError(f"{name}={reference!r} is not in model columns {self.models}")

    @property
    def model_order(self) -> list[str]:
        """Models sorted by overall score (best first).  Computed lazily."""
        if self._model_order is None:
            self._model_order = self.evaluate().sort_values().index.tolist()
        return self._model_order

    def _score_frame(
        self,
        cv: pd.DataFrame | None,
        *,
        keep_uids: bool,
        keep_metrics: bool,
    ) -> pd.Series | pd.DataFrame:
        cv_ = self.cv_df if cv is None else cv
        scores_df = uf_evaluate(
            df=cv_,
            models=self.models,
            metrics=self.metrics,
            train_df=self.train_df,
            id_col=self.id_col,
            time_col=self.time_col,
            target_col=self.target_col,
            cutoff_col=self.COLUMNS["cutoff"],
        )
        metric_col = self.COLUMNS["metric"]
        if keep_uids and keep_metrics:
            return scores_df.groupby([self.id_col, metric_col]).agg(
                self.agg_func, numeric_only=True
            )
        if keep_uids:
            return scores_df.groupby(self.id_col).agg(self.agg_func, numeric_only=True)
        if keep_metrics:
            return scores_df.groupby(metric_col).agg(self.agg_func, numeric_only=True)

        scores = scores_df.drop(columns=[self.id_col, metric_col]).agg(
            self.agg_func, numeric_only=True
        )
        scores.name = "Overall"
        return scores

    def uid_scores(
        self,
        cv: pd.DataFrame | None = None,
        keep_metrics: bool = False,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Per-unique-ID scores (index = unique IDs, columns = models).

        The full-frame, metric-averaged result is cached after the first call.
        """
        use_cache = cv is None and not keep_metrics
        if use_cache and self._uid_scores is not None and not refresh:
            return self._uid_scores

        scores = self._score_frame(cv, keep_uids=True, keep_metrics=keep_metrics)
        if use_cache:
            self._uid_scores = scores
        return scores

    def evaluate(
        self,
        cv: pd.DataFrame | None = None,
        keep_uids: bool = False,
        keep_metrics: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """Score models with the configured metrics.

        Parameters
        ----------
        cv : pd.DataFrame, optional
            Subset of the CV frame.  Defaults to the full ``cv_df``.
        keep_uids : bool, default False
            If True, return one row per unique ID; otherwise a Series of
            overall scores named ``"Overall"``.
        keep_metrics : bool, default False
            If True, do not average across metrics.  With ``keep_uids=True``
            the index is ``(unique_id, metric)``; otherwise rows are metrics.
        """
        if keep_uids:
            return self.uid_scores(cv=cv, keep_metrics=keep_metrics)

        if cv is None and not keep_metrics:
            scores = self.uid_scores().agg(self.agg_func, numeric_only=True)
            scores.name = "Overall"
            return scores

        return self._score_frame(cv, keep_uids=False, keep_metrics=keep_metrics)

    def evaluate_by_uid(
        self,
        cv: pd.DataFrame | None = None,
        keep_metrics: bool = False,
    ) -> pd.DataFrame:
        """Score models with one row per unique ID."""
        return self.evaluate(cv=cv, keep_uids=True, keep_metrics=keep_metrics)

    def evaluate_on_hard(self, cv: pd.DataFrame | None = None) -> pd.Series:
        """Mean (or median) error of each model on hard unique IDs."""
        return self.hardness.accuracy_on_hard(self.uid_scores(cv=cv))

    def expected_shortfall(self, cv: pd.DataFrame | None = None) -> pd.Series:
        """Expected shortfall (CVaR) of each model's per-UID errors."""
        return self.hardness.expected_shortfall(self.uid_scores(cv=cv))

    def hard_uids(self, cv: pd.DataFrame | None = None, return_df: bool = False):
        """Unique IDs whose reference error exceeds the hardness quantile."""
        return self.hardness.get_hard_uids(self.uid_scores(cv=cv), return_df=return_df)

    def winning_ratios(
        self,
        on_hard: bool = False,
        cv: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """ROPE win / draw / loss probabilities vs. the reference model."""
        err = self.uid_scores(cv=cv)
        if on_hard:
            err = self.hardness.get_hard_uids(err, return_df=True)
        return self.rope.get_winning_ratios(err)

    def aspect_table(
        self,
        cv: pd.DataFrame | None = None,
        group_cols: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Concatenate overall, tail, horizon-bound, and optional group scores.

        Returns
        -------
        pd.DataFrame
            Index is models; columns are aspect names.
        """
        parts: list[pd.Series | pd.DataFrame] = [
            self.evaluate(cv=cv),
            self.expected_shortfall(cv=cv),
            self.evaluate_by_horizon_bounds(cv=cv),
        ]
        if self.hardness.reference is not None:
            parts.append(self.evaluate_on_hard(cv=cv))
        if group_cols:
            for col in group_cols:
                parts.append(self.evaluate_by_group(col, cv=cv))
        return pd.concat(parts, axis=1)

    def evaluate_by_horizon(
        self,
        cv: pd.DataFrame | None = None,
        group_by_freq: bool = False,
        freq_col: str = "Frequency",
        cumulative: bool = True,
    ) -> pd.DataFrame:
        """Score models at each forecast horizon.

        Parameters
        ----------
        cumulative : bool, default True
            If True, score all observations with ``horizon <= h``.  If False,
            score only step ``h``.
        group_by_freq : bool, default False
            If True, compute a separate horizon curve per ``freq_col`` level.
        freq_col : str, default 'Frequency'
            Column used when ``group_by_freq`` is True.
        """
        cv_ = self.cv_df if cv is None else cv
        if group_by_freq and freq_col not in cv_.columns:
            raise KeyError(f"Column '{freq_col}' not found in DataFrame")

        cv_groups = cv_.groupby(freq_col) if group_by_freq else [(None, cv_)]

        scores_by_group = {}
        h_col = self.COLUMNS["horizon"]
        for g, df in cv_groups:
            fh = df[h_col].sort_values().unique()
            if cumulative:
                eval_horizon = {h: self.evaluate(df.loc[df[h_col] <= h]) for h in fh}
            else:
                eval_horizon = {h: self.evaluate(df.loc[df[h_col] == h]) for h in fh}
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
        h_col = self.COLUMNS["horizon"]
        h_by_uid = cv_.groupby(self.id_col)[h_col]
        first_horizon = cv_.loc[cv_[h_col].eq(h_by_uid.transform("min"))]
        last_horizon = cv_.loc[cv_[h_col].eq(h_by_uid.transform("max"))]

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
    ) -> pd.DataFrame:
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
        if anomaly_col not in cv_.columns:
            raise KeyError(f"Column '{anomaly_col}' not found in DataFrame")

        has_anomaly = cv_.groupby(self.id_col)[anomaly_col].transform("sum") > 0
        flagged = cv_.loc[has_anomaly]
        if mode == "observations":
            flagged = flagged.loc[flagged[anomaly_col] > 0]
        if flagged.empty:
            return pd.DataFrame(columns=self.models)

        return self.evaluate(flagged, keep_uids=True)

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
