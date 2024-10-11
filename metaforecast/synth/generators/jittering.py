import pandas as pd
import numpy as np

from metaforecast.synth.generators._base import SemiSyntheticTransformer


class Jittering(SemiSyntheticTransformer):
    """ Jittering

    Adding Gaussian noise to time series

    References:
        Um, T. T., Pfister, F. M., Pichler, D., Endo, S., Lang, M., Hirche, S., ... & Kulić, D. (2017, November).
        Data augmentation of wearable sensor data for parkinson’s disease monitoring using convolutional neural
        networks. In Proceedings of the 19th ACM international conference on multimodal interaction (pp. 216-220).

    Example usage:
    >>> import pandas as pd
    >>> from datasetsforecast.m3 import M3
    >>> from neuralforecast import NeuralForecast
    >>> from neuralforecast.models import NHITS
    >>>
    >>> from metaforecast.synth import Jittering
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
    >>> tsgen = Jittering(sigma=0.5)
    >>>
    >>> # Add jittering to each time series
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

    def __init__(self, sigma: float = 0.03, rename_uids: bool = True):
        """
        :param sigma: Scaling parameter for Gaussian noise
        :type sigma: float. Defaults to 0.03

        :param rename_uids: whether to rename the original unique_id's
        :type rename_uids: bool
        """
        super().__init__(alias='JITTER', rename_uids=rename_uids)

        self.sigma = sigma

    def _create_synthetic_ts(self, df: pd.DataFrame) -> pd.DataFrame:
        df_ = df.copy()

        # todo or standardize first?
        sig = self.sigma * df_['y'].std()

        df_.loc[:, 'y'] += np.random.normal(loc=0., scale=sig, size=df_.shape[0])

        return df_
