from __future__ import annotations

import pandas as pd


class ActiveTesting:
    """Greedy active-testing selector over a configuration score matrix.

    Starts from the config with the best mean score, then iteratively
    picks the remaining config with the highest probability of beating
    the current champion (fraction of samples where it wins).  Optional
    filters drop highly correlated configs, require a minimum mean
    improvement to update the champion, and stop early when no candidate
    has a realistic chance.

    Parameters
    ----------
    use_ranks : bool, default False
        Convert scores to per-sample ranks before comparisons.
    max_trials : int, optional
        Maximum number of configs to select.  ``None`` selects until
        the candidate pool is empty (or an early-stop rule fires).
    corr_threshold : float, optional
        Skip a candidate if its absolute correlation with any already
        selected config exceeds this value (e.g. ``0.9``).
    delta : float, optional
        Minimum mean-score improvement required to replace the champion.
        ``None`` treats any strict improvement as enough.
    min_win_prob : float, optional
        Stop if the best remaining candidate's win probability falls
        below this value.
    lower_is_better : bool, default True
        If True, lower scores are better (error metrics).  If False,
        higher scores are better (accuracy).

    Attributes
    ----------
    selected_ : list
        Ordered config IDs from the last :meth:`select` call.
    path_ : pd.DataFrame
        Per-step diagnostics from the last :meth:`select` call, with
        columns ``step``, ``config``, ``win_prob``, ``mean_score``,
        ``became_champion``.

    Examples
    --------
    >>> import pandas as pd
    >>> from metaforecast.coseal import ActiveTesting
    >>>
    >>> scores = pd.DataFrame({
    ...     "cfg_a": [0.4, 0.5, 0.3],
    ...     "cfg_b": [0.2, 0.6, 0.4],
    ...     "cfg_c": [0.5, 0.4, 0.2],
    ... })
    >>> selector = ActiveTesting(max_trials=2)
    >>> order = selector.select(scores)
    >>> order  # doctest: +SKIP
    """

    def __init__(
        self,
        use_ranks: bool = False,
        max_trials: int | None = None,
        corr_threshold: float | None = None,
        delta: float | None = None,
        min_win_prob: float | None = None,
        lower_is_better: bool = True,
    ):
        self.use_ranks = use_ranks
        self.max_trials = max_trials
        self.corr_threshold = corr_threshold
        self.delta = delta
        self.min_win_prob = min_win_prob
        self.lower_is_better = lower_is_better
        self.selected_: list | None = None
        self.path_: pd.DataFrame | None = None

    def select(self, scores_df: pd.DataFrame) -> list:
        """Order configurations by active-testing priority.

        Parameters
        ----------
        scores_df : pd.DataFrame
            Rows are samples (e.g. ``unique_id``), columns are config IDs,
            values are scores.

        Returns
        -------
        list
            Config IDs in the order they were selected.
        """
        if scores_df.shape[1] == 0:
            raise ValueError("scores_df must contain at least one configuration.")

        df = scores_df.copy()
        if self.use_ranks:
            df = df.rank(axis=1, method="average", ascending=self.lower_is_better)

        remaining = set(df.columns)
        selected: list = []
        path: list[dict] = []
        selected_corr_cache: dict = {}

        means = df.mean()
        best_config = means.idxmin() if self.lower_is_better else means.idxmax()
        selected.append(best_config)
        remaining.remove(best_config)
        path.append(
            {
                "step": 0,
                "config": best_config,
                "win_prob": float("nan"),
                "mean_score": float(means[best_config]),
                "became_champion": True,
            }
        )

        if self.corr_threshold is not None:
            selected_corr_cache[best_config] = df[best_config]

        while remaining:
            if self.max_trials is not None and len(selected) >= self.max_trials:
                break

            best_scores = df[best_config]
            candidate_cols = list(remaining)

            if self.corr_threshold is not None:
                filtered_cols = []
                for col in candidate_cols:
                    dominated = any(
                        abs(df[col].corr(selected_corr_cache[sel])) > self.corr_threshold
                        for sel in selected
                    )
                    if not dominated:
                        filtered_cols.append(col)
                candidate_cols = filtered_cols

            if not candidate_cols:
                break

            candidates_df = df[candidate_cols]
            if self.lower_is_better:
                win_probs = candidates_df.lt(best_scores, axis=0).mean()
            else:
                win_probs = candidates_df.gt(best_scores, axis=0).mean()

            best_prob = float(win_probs.max())
            best_candidate = win_probs.idxmax()

            if self.min_win_prob is not None and best_prob < self.min_win_prob:
                break

            selected.append(best_candidate)
            remaining.remove(best_candidate)

            if self.corr_threshold is not None:
                selected_corr_cache[best_candidate] = df[best_candidate]

            candidate_mean = means[best_candidate]
            best_mean = means[best_config]
            if self.lower_is_better:
                improvement = best_mean - candidate_mean
            else:
                improvement = candidate_mean - best_mean

            threshold = self.delta if self.delta is not None else 0.0
            became_champion = improvement > threshold
            if became_champion:
                best_config = best_candidate

            path.append(
                {
                    "step": len(selected) - 1,
                    "config": best_candidate,
                    "win_prob": best_prob,
                    "mean_score": float(candidate_mean),
                    "became_champion": became_champion,
                }
            )

        self.selected_ = selected
        self.path_ = pd.DataFrame(path)
        return selected
