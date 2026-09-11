import numpy as np
import pandas as pd

from metaforecast.synth.generators.base import SemiSyntheticGenerator


class TSMixup(SemiSyntheticGenerator):
    """Generate synthetic time series using weighted averaging of multiple series.

    Creates new time series by computing weighted combinations of existing
    series, inspired by image Mixup and adapted for time series in
    Chronos. This method preserves temporal characteristics while
    creating diverse, realistic variations.

    References
    ----------
    Ansari, A. F., et al. (2024). "Chronos: Learning the language of time series."
    arXiv preprint arXiv:2403.07815.

    Examples
    --------
    >>> import pandas as pd
    >>> from datasetsforecast.m3 import M3
    >>> from neuralforecast import NeuralForecast
    >>> from neuralforecast.models import NHITS
    >>>
    >>> from metaforecast.synth import TSMixup
    >>> from metaforecast.evaluation.cv import SeriesWiseSplit
    >>>
    >>>
    >>> # Loading and preparing data
    >>> df, *_ = M3.load('.', group='Monthly')
    >>>
    >>> horizon = 12
    >>> train, test = SeriesWiseSplit.time_wise_split(df, horizon)
    >>>
    >>> # Data augmentation
    >>> tsgen = TSMixup(min_len=50, max_len=96, max_n_uids=7)
    >>> # Applying time warping to each time series in the dataset
    >>> synth_df = tsgen.transform(train)
    >>>
    >>> # Concat the synthetic dataset with the original training data
    >>> train_aug = pd.concat([train, synth_df])
    >>>
    >>> # Setting up NHITS
    >>> models = [NHITS(input_size=horizon, h=horizon, accelerator='cpu')]
    >>> nf = NeuralForecast(models=models, freq='M')
    >>>
    >>> # Fitting NHITS on the augmented data
    >>> nf.fit(df=train_aug)
    >>>
    >>> # Forecasting on the original dataset
    >>> fcst = nf.predict(df=train)
    """

    def __init__(
        self,
        max_n_uids: int,
        min_len: int,
        max_len: int,
        dirichlet_alpha: float = 1.5,
    ):
        """Initialize TSMixup transformer with mixing parameters.

        Parameters
        ----------
        max_n_uids : int
            Maximum number of source series to combine for each synthetic series:
            - Higher values allow more complex combinations
            - Lower values create simpler mixtures
            - Must be ≥ 2 to enable mixing
            Controls diversity of generated patterns.

        min_len : int
            Minimum length of generated series in observations.
            Must satisfy: min_len ≤ max_len
            Useful for creating variable-length datasets.

        max_len : int
            Maximum length of generated series in observations.
            Must satisfy: max_len ≥ min_len
            Controls upper bound of series length.

        dirichlet_alpha : float, default=1.5
            Concentration parameter for Dirichlet distribution
            used to generate mixing weights:

        """
        super().__init__(alias="TSMixup")

        self.min_len = min_len
        self.max_len = max_len
        self.max_n_uids = max_n_uids
        self.dirichlet_alpha = dirichlet_alpha

    def transform(self, df: pd.DataFrame, n_series: int = -1, **kwargs):
        """Apply TSMixup to create synthetic time series variations.

        Generates new time series by computing weighted combinations of
        existing series. Each synthetic series combines up to max_n_uids
        source series using Dirichlet-distributed weights.

        Parameters
        ----------
        df : pd.DataFrame
            Source time series dataset with required columns:
            - unique_id: Series identifier
            - ds: Timestamp
            - y: Target values
            Must follow nixtla framework conventions

        n_series : int, default=-1
            Number of synthetic series to generate:
            - If -1: Generate one per input series
            - If positive: Generate specified number

        Returns
        -------
        pd.DataFrame
            Generated synthetic series with columns:
            - unique_id: f"mixup_{i}" for i in range(n_series)
            - ds: Timestamps (length between min_len and max_len)
            - y: Mixed values from source series

        """
        self._assert_datatypes(df)

        y_by_uid, ds_by_uid, lengths = self._index_series(df)
        n_pool = len(y_by_uid)
        if n_pool == 0:
            raise ValueError("df contains no series to mix")

        if n_series < 0:
            n_series = n_pool

        uid_chunks = []
        ds_chunks = []
        y_chunks = []
        for _ in range(n_series):
            n_uids = np.random.randint(1, self.max_n_uids + 1)
            chosen = np.unique(np.random.choice(n_pool, n_uids, replace=True))
            ds, y = self._mix_series(y_by_uid, ds_by_uid, lengths, chosen)
            uid_chunks.append(np.full(len(y), f"{self.alias}_{self.counter}", dtype=object))
            ds_chunks.append(ds)
            y_chunks.append(y)
            self.counter += 1

        return pd.DataFrame(
            {
                self.id_col: np.concatenate(uid_chunks),
                self.time_col: np.concatenate(ds_chunks),
                self.target_col: np.concatenate(y_chunks),
            }
        )

    def _index_series(self, df: pd.DataFrame):
        """Split the panel into per-series NumPy arrays (one groupby)."""
        y_by_uid = []
        ds_by_uid = []
        lengths = []
        for _, uid_df in df.groupby(self.id_col, sort=False):
            y_by_uid.append(uid_df[self.target_col].to_numpy())
            ds_by_uid.append(uid_df[self.time_col].to_numpy())
            lengths.append(len(uid_df))
        return y_by_uid, ds_by_uid, np.asarray(lengths)

    def _create_synthetic_ts(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        y_by_uid, ds_by_uid, lengths = self._index_series(df)
        ds, y = self._mix_series(y_by_uid, ds_by_uid, lengths, np.arange(len(y_by_uid)))
        return pd.DataFrame({self.time_col: ds, self.target_col: y})

    def _mix_series(self, y_by_uid, ds_by_uid, lengths, chosen: np.ndarray):
        """Weighted mix of the series indexed by ``chosen``."""
        smallest_n = int(lengths[chosen].min())
        max_len_ = min(smallest_n, self.max_len)
        min_len_ = min(smallest_n, self.min_len)

        if self.min_len == self.max_len:
            n_obs = min_len_
        elif max_len_ < min_len_:
            n_obs = max_len_
        else:
            n_obs = int(np.random.randint(min_len_, max_len_ + 1))

        weights = self.sample_weights_dirichlet(self.dirichlet_alpha, len(chosen))
        y = np.zeros(n_obs, dtype=float)
        for weight, idx in zip(weights, chosen, strict=True):
            start = int(np.random.randint(0, lengths[idx] - n_obs + 1))
            y += weight * y_by_uid[idx][start : start + n_obs]

        return ds_by_uid[chosen[0]][:n_obs], y
