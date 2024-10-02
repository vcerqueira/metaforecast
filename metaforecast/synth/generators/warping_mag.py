from scipy.interpolate import CubicSpline
import pandas as pd
import numpy as np

from metaforecast.synth.generators._base import SemiSyntheticTransformer


class MagnitudeWarping(SemiSyntheticTransformer):

    def __init__(self, sigma: float = 0.2, knot=4, rename_uids: bool=True):
        super().__init__(alias='MWARP', rename_uids=rename_uids)

        self.sigma = sigma
        self.knot = knot

    def _create_synthetic_ts(self, df: pd.DataFrame) -> pd.DataFrame:
        df_ = df.copy()

        warper = self.get_warper(df_.loc[:, 'y'].values)

        df_.loc[:, 'y'] *= warper

        return df_

    def get_warper(self, x: np.ndarray):
        x = x.reshape(-1, 1)

        orig_steps = np.arange(x.shape[0])

        random_warps = np.random.normal(
            loc=1.0, scale=self.sigma, size=(self.knot + 2, x.shape[1])
        )

        warp_steps = np.linspace(0, x.shape[0] - 1.0, num=self.knot + 2)
        warper = np.zeros((x.shape[0], x.shape[1]))

        for i in range(x.shape[1]):
            warper[:, i] = np.array(
                [CubicSpline(warp_steps, random_warps[:, i])(orig_steps)]
            )

        warper = warper.squeeze()

        return warper
