"""Series-wise bootstrap CV splitters."""

from __future__ import annotations

import numpy as np

from metaforecast.evaluation.cv._base import SeriesWiseSplit


class SeriesWiseBootstrap(SeriesWiseSplit):
    """Single bootstrap split over unique IDs.

    Samples *n* series with replacement for training; the out-of-bag
    series form the test set.

    Parameters
    ----------
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(self, random_state: int | None = None):
        super().__init__(n_splits=1)
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        rng = np.random.RandomState(self.random_state)
        n = len(X)
        indices = np.arange(n)
        train_idx = rng.choice(indices, size=n, replace=True)
        test_idx = np.setdiff1d(indices, np.unique(train_idx))
        yield train_idx, test_idx


class SeriesWiseRepeatedBootstrap(SeriesWiseSplit):
    """Repeated bootstrap splits over unique IDs.

    Parameters
    ----------
    n_repeats : int
        Number of bootstrap repetitions.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(self, n_repeats: int, random_state: int | None = None):
        super().__init__(n_splits=n_repeats)
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        rng = np.random.RandomState(self.random_state)
        n = len(X)
        indices = np.arange(n)
        for _ in range(self.n_repeats):
            train_idx = rng.choice(indices, size=n, replace=True)
            test_idx = np.setdiff1d(indices, np.unique(train_idx))
            yield train_idx, test_idx
