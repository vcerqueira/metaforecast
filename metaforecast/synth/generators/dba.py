import numpy as np
import pandas as pd

from tslearn.barycenters import dtw_barycenter_averaging_subgradient as dtw

from metaforecast.synth.generators.base import SemiSyntheticGenerator


class DBA(SemiSyntheticGenerator):
    """ DBA

    DTW Barycentric Averaging synthetic time series generator.

    References:
        Forestier, G., Petitjean, F., Dau, H.A., Webb, G.I., Keogh, E.: Generating synthetic
        time series to augment sparse datasets. In: 2017 IEEE international conference on data
        mining (ICDM), pp. 865–870. IEEE (2017)

    Attributes:
        DTW_PARAMS (Dict[str, float]) DTW configuration parameters

    Example usage (CHECK NOTEBOOKS FOR MORE EXTENDED EXAMPLES):
    >>> import pandas as pd
    >>> from datasetsforecast.m3 import M3
    >>> from neuralforecast import NeuralForecast
    >>> from neuralforecast.models import NHITS
    >>>
    >>> from metaforecast.synth import DBA
    >>> from metaforecast.utils.data import DataUtils
    >>>
    >>> # Loading and preparing data
    >>> df, *_ = M3.load('.', group='Monthly')
    >>>
    >>> horizon = 12
    >>> train, test = DataUtils.train_test_split(df, horizon)
    >>>
    >>> # Data augmentation
    >>> tsgen = DBA(max_n_uids=10)
    >>> ## Create 100 time series
    >>> synth_df = tsgen.transform(train, 100)
    >>> ## Concat the synthetic dataset with the original training data
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

    DTW_PARAMS = {'max_iter': 10, 'tol': 1e-3}

    def __init__(self, max_n_uids: int, dirichlet_alpha: float = 1.0):
        """
        :param max_n_uids: Maximum number of time series (unique_id's) to use in a given generation
        operation
        :type max_n_uids: int

        :param dirichlet_alpha: Gamma distribution alpha parameter value for weighting the selected
        time series.
        :type dirichlet_alpha: float. Default = 1.0
        """
        super().__init__(alias='DBA')

        self.max_n_uids = max_n_uids
        self.dirichlet_alpha = dirichlet_alpha

    # pylint: disable=unused-variable
    def transform(self, df: pd.DataFrame, n_series: int = -1, **kwargs):
        """ transform

        Generate synthetic time series based on a source df using DBA

        :param df: time series dataset with unique_id, ds, y columns following a nixtla-based
        structure
        :type df: pd.DataFrame

        :param n_series: Number of series to generate
        :type n_series: int. Defaults to -1, which means creating a number of time series
        equal to the number of time series in the source dataset

        """
        self._assert_datatypes(df)

        unq_uids = df['unique_id'].unique()

        if n_series < 0:
            n_series = len(unq_uids)

        dataset = []
        for _ in range(n_series):
            n_uids = np.random.randint(1, self.max_n_uids + 1)

            selected_uids = np.random.choice(unq_uids, n_uids, replace=False).tolist()

            df_uids = df.query('unique_id == @selected_uids')

            ts_df = self._create_synthetic_ts(df_uids)
            ts_df['unique_id'] = f'{self.alias}_{self.counter}'
            self.counter += 1

            dataset.append(ts_df)

        synth_df = pd.concat(dataset).reset_index(drop=True)

        return synth_df

    # pylint: disable=arguments-differ
    def _create_synthetic_ts(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """ _create_synthetic_ts

        Apply DBA to a time series dataset

        :param df: time series dataset with a sample of unique_id's
        :return: pd.DataFrame with synthetic time series
        """
        y_list = [y['y'].values for _, y in df.groupby('unique_id')]
        uid_size = df['unique_id'].value_counts()

        ds = df.query(f'unique_id=="{uid_size.index[0]}"')['ds'].values

        w = self.sample_weights_dirichlet(1, len(y_list))

        synth_y = dtw(X=y_list, weights=w, **self.DTW_PARAMS)
        synth_y = synth_y.flatten()

        synth_df = pd.DataFrame({'ds': ds[:len(synth_y)], 'y': synth_y})

        return synth_df
