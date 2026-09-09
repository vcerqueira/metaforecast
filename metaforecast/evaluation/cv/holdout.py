"""Series-wise holdout and Monte Carlo CV splitters."""

from __future__ import annotations

import numpy as np

from metaforecast.evaluation.cv._base import SeriesWiseSplit


class SeriesWiseHoldout(SeriesWiseSplit):
    """Single random train/test split over unique IDs.

    Parameters
    ----------
    train_size : float
        Fraction of series to use for training (0, 1).
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(self, train_size: float, random_state: int | None = None):
        super().__init__(n_splits=1)
        self.train_size = train_size
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        rng = np.random.RandomState(self.random_state)
        n = len(X)
        n_train = int(n * self.train_size)
        indices = np.arange(n)
        train_idx = rng.choice(indices, size=n_train, replace=False)
        test_idx = np.setdiff1d(indices, train_idx)
        yield train_idx, test_idx


class SeriesWiseRepeatedHoldout(SeriesWiseSplit):
    """Repeated random train/test splits over unique IDs.

    Parameters
    ----------
    train_size : float
        Fraction of series for training.
    n_repeats : int
        Number of repetitions.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        train_size: float,
        n_repeats: int,
        random_state: int | None = None,
    ):
        super().__init__(n_splits=n_repeats)
        self.train_size = train_size
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        rng = np.random.RandomState(self.random_state)
        n = len(X)
        n_train = int(n * self.train_size)
        indices = np.arange(n)
        for _ in range(self.n_repeats):
            train_idx = rng.choice(indices, size=n_train, replace=False)
            test_idx = np.setdiff1d(indices, train_idx)
            yield train_idx, test_idx


class SeriesWiseMonteCarlo(SeriesWiseSplit):
    """Monte Carlo series-wise CV: random disjoint train and test subsets.

    Unlike :class:`SeriesWiseRepeatedHoldout`, the test set is also a
    *subset* of the non-training series (some series are unused per fold).

    Parameters
    ----------
    train_size : float
        Fraction of series for training.
    test_size : float
        Fraction of series for testing (must satisfy
        ``train_size + test_size < 1``).
    n_repeats : int
        Number of repetitions.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        train_size: float,
        test_size: float,
        n_repeats: int,
        random_state: int | None = None,
    ):
        if train_size + test_size >= 1.0:
            raise ValueError("train_size + test_size must be less than 1.0")
        super().__init__(n_splits=n_repeats)
        self.train_size = train_size
        self.test_size = test_size
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        rng = np.random.RandomState(self.random_state)
        n = len(X)
        n_train = int(n * self.train_size)
        n_test = int(n * self.test_size)
        indices = np.arange(n)
        for _ in range(self.n_repeats):
            selected = rng.choice(indices, size=n_train + n_test, replace=False)
            train_idx = rng.choice(selected, size=n_train, replace=False)
            train_set = set(train_idx)
            test_idx = np.array([i for i in selected if i not in train_set])
            yield train_idx, test_idx
