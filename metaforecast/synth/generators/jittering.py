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
