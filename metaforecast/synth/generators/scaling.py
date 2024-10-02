import pandas as pd
import numpy as np

from metaforecast.synth.generators._base import SemiSyntheticTransformer


class TSscaling(SemiSyntheticTransformer):

    def __init__(self, sigma: float = 0.1, rename_uids: bool=True):
        super().__init__(alias='SCALE', rename_uids=rename_uids)

        self.sigma = sigma

    def _create_synthetic_ts(self, df: pd.DataFrame) -> pd.DataFrame:
        df_ = df.copy()

        factor = np.random.normal(loc=1.0, scale=self.sigma, size=df_.shape[0])

        df_.loc[:, 'y'] *= factor

        return df_
