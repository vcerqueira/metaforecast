import numpy as np
import pandas as pd

from metaforecast.synth.generators._base import SemiSyntheticGenerator


class TSMixup(SemiSyntheticGenerator):
    """ TSMixup

    Synthetic time series generation based on weighted averages of several time series

    References:
        Ansari, A. F., Stella, L., Turkmen, C., Zhang, X., Mercado, P., Shen, H., ... & Wang, Y. (2024).
        Chronos: Learning the language of time series. arXiv preprint arXiv:2403.07815.

    Example usage (check notebooks for extended examples):
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

    def __init__(self,
                 max_n_uids: int,
                 min_len: int,
                 max_len: int,
                 dirichlet_alpha: float = 1.5):
        """

        :param max_n_uids: Maximum number of time series (unique_id's) to consider for generating a given
        time series
        :type max_n_uids: int

        :param min_len: Minimum number of observations of the new synthetic time series
        :param min_len: int

        :param max_len: Maximum number of observations of the new synthetic time series
        :param max_len: int

        :param dirichlet_alpha: Alpha parameter for the Gamma distribution
        :type dirichlet_alpha: float. Defaults to 1.5
        """

        super().__init__(alias='TSMixup')

        self.min_len = min_len
        self.max_len = max_len
        self.max_n_uids = max_n_uids
        self.dirichlet_alpha = dirichlet_alpha

    def transform(self, df: pd.DataFrame, n_series: int = -1):
        self._assert_datatypes(df)

        unq_uids = df['unique_id'].unique()

        if n_series < 0:
            n_series = len(unq_uids)

        dataset = []
        for i in range(n_series):
            n_uids = np.random.randint(1, self.max_n_uids + 1)

            selected_uids = np.random.choice(unq_uids, n_uids, replace=False).tolist()

            df_uids = df.query('unique_id == @selected_uids')

            ts_df = self._create_synthetic_ts(df_uids)
            ts_df['unique_id'] = f'{self.alias}_{self.counter}'
            self.counter += 1

            dataset.append(ts_df)

        synth_df = pd.concat(dataset).reset_index(drop=True)

        return synth_df

    def _create_synthetic_ts(self, df: pd.DataFrame) -> pd.DataFrame:
        uids = df['unique_id'].unique()

        smallest_n = df['unique_id'].value_counts().min()

        if smallest_n < self.max_len:
            max_len_ = smallest_n
        else:
            max_len_ = self.max_len

        if self.min_len == self.max_len:
            n_obs = self.min_len
        else:
            n_obs = np.random.randint(self.min_len, max_len_ + 1)

        w = self.sample_weights_dirichlet(self.dirichlet_alpha, len(uids))

        ds = df.query(f'unique_id=="{uids[0]}"').head(n_obs)['ds'].values

        mixup = []
        for j, k in enumerate(uids):
            uid_df = df.query(f'unique_id=="{k}"').head(n_obs)

            uid_y = uid_df['y'].reset_index(drop=True)
            uid_y /= uid_y.mean()
            uid_y *= w[j]

            mixup.append(uid_y)

        y = pd.concat(mixup, axis=1).sum(axis=1).values

        synth_df = pd.DataFrame({'ds': ds, 'y': y, })

        return synth_df
