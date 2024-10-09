import pandas as pd
import numpy as np

from metaforecast.synth.generators._base import SemiSyntheticTransformer


class Jittering(SemiSyntheticTransformer):
    # https://github.com/luisroque/robustness_hierarchical_time_series_forecasting_algorithms/blob/main/tsaugmentation/transformations/manipulate_data.py
    # https://github.com/uchidalab/time_series_augmentation/blob/master/utils/augmentation.py

    def __init__(self, sigma: float = 0.03, rename_uids: bool = True):
        super().__init__(alias='JITTER', rename_uids=rename_uids)

        self.sigma = sigma

    def _create_synthetic_ts(self, df: pd.DataFrame) -> pd.DataFrame:
        df_ = df.copy()

        # todo or standardize first?
        sig = self.sigma * df_['y'].std()

        df_.loc[:, 'y'] += np.random.normal(loc=0., scale=sig, size=df_.shape[0])

        return df_
