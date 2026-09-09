"""Region of Practical Equivalence (ROPE) analysis for model comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd


class RopeAnalysis:
    """Compare models against a reference using a ROPE threshold.

    Percentage differences vs. the reference are binned into three
    outcomes: the reference loses, a draw (within ``±rope`` percent),
    or the reference wins.

    Parameters
    ----------
    rope : float
        Region of practical equivalence as a percentage.  Differences
        inside ``±rope`` count as draws.
    reference : str
        Column name of the reference model in the per-UID score matrix.
    """

    SIDES = ["{reference} loses", "draw", "{reference} wins"]

    def __init__(self, rope: float, reference: str | None):
        self.rope = rope
        self.reference = reference
        self.sides = [side.format(reference=reference) for side in self.SIDES]

    def get_winning_ratios(self, uid_scores: pd.DataFrame) -> pd.DataFrame:
        """Win / draw / loss probabilities vs. the reference, per model.

        Parameters
        ----------
        uid_scores : pd.DataFrame
            Per-series scores (index = unique IDs, columns = models).
            Must include the reference model.

        Returns
        -------
        pd.DataFrame
            Rows are models (reference excluded), columns are the three
            ROPE outcomes.
        """
        self._assert_params(uid_scores)
        scores_pd = self._calc_percentage_diff(uid_scores)
        prob_df = scores_pd.apply(self._calc_vector_side_probs, axis=0).T
        prob_df.columns = self.sides
        return prob_df

    def _calc_vector_side_probs(self, diff_vec: pd.Series) -> tuple[float, float, float]:
        left = float((diff_vec < -self.rope).mean())
        right = float((diff_vec > self.rope).mean())
        mid = float(np.mean([-self.rope < x_ < self.rope for x_ in diff_vec]))
        return left, mid, right

    def _calc_percentage_diff(self, scores: pd.DataFrame) -> pd.DataFrame:
        scores_pd = {
            mod: self._percentage_diff(scores[mod], scores[self.reference])
            for mod in scores.columns
            if mod != self.reference
        }
        return pd.DataFrame(scores_pd, index=scores.index)

    def _assert_params(self, scores: pd.DataFrame) -> None:
        if self.reference not in scores.columns:
            raise ValueError(f"{self.reference} not in scores columns")

    @staticmethod
    def _percentage_diff(x, y):
        return ((x - y) / abs(y)) * 100
