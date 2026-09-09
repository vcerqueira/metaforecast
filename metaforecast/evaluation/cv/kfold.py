"""Series-wise K-Fold CV splitters."""

from __future__ import annotations

from sklearn.model_selection import KFold, RepeatedKFold

from metaforecast.evaluation.cv._base import SeriesWiseSplit


class SeriesWiseKFold(SeriesWiseSplit):
    """K-Fold cross-validation over unique IDs.

    Parameters
    ----------
    n_splits : int, default 5
        Number of folds.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(self, n_splits: int = 5, random_state: int | None = None):
        super().__init__(n_splits=n_splits)
        self._kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def split(self, X, y=None, groups=None):
        yield from self._kf.split(X, y, groups)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self._kf.get_n_splits(X, y, groups)


class SeriesWiseRepeatedKFold(SeriesWiseSplit):
    """Repeated K-Fold cross-validation over unique IDs.

    Parameters
    ----------
    n_splits : int, default 5
        Number of folds per repeat.
    n_repeats : int, default 2
        Number of times to repeat the K-Fold.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_repeats: int = 2,
        random_state: int | None = None,
    ):
        super().__init__(n_splits=n_splits * n_repeats)
        self._rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    def split(self, X, y=None, groups=None):
        yield from self._rkf.split(X, y, groups)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self._rkf.get_n_splits(X, y, groups)
