from scipy.interpolate import CubicSpline
import pandas as pd
import numpy as np

from metaforecast.synth.generators._base import SemiSyntheticTransformer


class TimeWarping(SemiSyntheticTransformer):

    def __init__(self, sigma: float = 0.2, knot=4, rename_uids: bool=True):
        super().__init__(alias='TWARP', rename_uids=rename_uids)

        self.sigma = sigma
        self.knot = knot

    def _create_synthetic_ts(self, df: pd.DataFrame) -> pd.DataFrame:
        df_ = df.copy()

        df_.loc[:, 'y'] = self.apply_time_warping(df_.loc[:, 'y'].values)

        return df_

    def apply_time_warping(self, x: np.ndarray):
        x = x.reshape(-1, 1)

        orig_steps = np.arange(x.shape[0])

        random_warps = np.random.normal(
            loc=1.0, scale=self.sigma, size=(self.knot + 2, x.shape[1])
        )
        warp_steps = np.linspace(0, x.shape[0] - 1.0, num=self.knot + 2)
        time_warp = np.zeros((x.shape[0], x.shape[1]))
        x_warped = np.zeros((x.shape[0], x.shape[1]))

        for i in range(x.shape[1]):
            time_warp[:, i] = CubicSpline(warp_steps, warp_steps * random_warps[:, i])(
                orig_steps
            )
            x_warped[:, i] = np.interp(orig_steps, time_warp[:, i], x[:, i])

        x_warped = x_warped.squeeze()

        return x_warped
