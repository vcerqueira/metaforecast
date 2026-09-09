"""Base class for series-wise cross-validation splitters.

All splitters operate on the *series* (unique_id) dimension: in each
fold, a subset of series is used for training and a disjoint subset for
testing.  The temporal split (history vs. future) is handled separately
by the forecasting framework.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd
from sklearn.model_selection._split import BaseCrossValidator


class SeriesWiseSplit(BaseCrossValidator, ABC):
    """Abstract base for series-wise CV splitters.

    Parameters
    ----------
    n_splits : int or None
        Number of folds.
    """

    def __init__(self, n_splits: int | None = None):
        self.n_splits = n_splits

    @abstractmethod
    def split(self, X, y=None, groups=None):
        """Yield ``(train_indices, test_indices)`` over unique IDs."""

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    @staticmethod
    def time_wise_split(
        df: pd.DataFrame,
        horizon: int,
        id_col: str = "unique_id",
        time_col: str = "ds",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split each series into train (history) and test (last *h* steps).

        Parameters
        ----------
        df : pd.DataFrame
        horizon : int
        id_col, time_col : str

        Returns
        -------
        train_df, test_df : pd.DataFrame
        """
        train_l, test_l = [], []
        for _, df_ in df.groupby(id_col):
            df_ = df_.sort_values(time_col)
            train_l.append(df_.head(-horizon))
            test_l.append(df_.tail(horizon))

        train_df = pd.concat(train_l).reset_index(drop=True)
        test_df = pd.concat(test_l).reset_index(drop=True)
        return train_df, test_df
