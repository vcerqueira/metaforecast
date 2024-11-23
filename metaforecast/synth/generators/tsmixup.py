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
    >>> from metaforecast.utils.data import DataUtils
    >>>
    >>>
    >>> # Loading and preparing data
    >>> df, *_ = M3.load('.', group='Monthly')
    >>>
    >>> horizon = 12
    >>> train, test = DataUtils.train_test_split(df, horizon)
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
        super().__init__(alias='TSMixup')

        self.min_len = min_len
        self.max_len = max_len
        self.max_n_uids = max_n_uids
        self.dirichlet_alpha = dirichlet_alpha

    # pylint: disable=arguments-differ
    # pylint: disable=unused-variable
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

        unq_uids = df['unique_id'].unique()

        if n_series < 0:
            n_series = len(unq_uids)

        dataset = []
        for _ in range(n_series):
            n_uids = np.random.randint(1, self.max_n_uids + 1)

            selected_uids = np.random.choice(unq_uids, n_uids, replace=True).tolist()

            df_uids = df.query('unique_id == @selected_uids')

            ts_df = self._create_synthetic_ts(df_uids)
            ts_df['unique_id'] = f"{self.alias}_{self.counter}"
            self.counter += 1

            dataset.append(ts_df)

        synth_df = pd.concat(dataset).reset_index(drop=True)

        return synth_df

    # pylint: disable=arguments-differ
    def _create_synthetic_ts(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:

        uids = df['unique_id'].unique()

        smallest_n = df['unique_id'].value_counts().min()

        max_len_ = smallest_n if smallest_n < self.max_len else self.max_len
        min_len_ = smallest_n if smallest_n < self.min_len else self.min_len

        if self.min_len == self.max_len:
            n_obs = min_len_
        elif max_len_ < min_len_:
            n_obs = max_len_
        else:
            n_obs = np.random.randint(min_len_, max_len_ + 1)

        w = self.sample_weights_dirichlet(self.dirichlet_alpha, len(uids))

        ds = df.query(f'unique_id=="{uids[0]}"').head(n_obs)['ds'].values

        mixup = []
        for j, k in enumerate(uids):
            df_j = df.query(f'unique_id=="{k}"')

            start_idx = np.random.randint(0, df_j.shape[0] - n_obs + 1)

            uid_df = df_j.iloc[start_idx : start_idx + n_obs]

            uid_y = uid_df['y'].reset_index(drop=True)

            uid_y *= w[j]

            mixup.append(uid_y)

        y = pd.concat(mixup, axis=1).sum(axis=1).values

        synth_df = pd.DataFrame(
            {
                'ds': ds,
                'y': y,
            }
        )

        return synth_df
