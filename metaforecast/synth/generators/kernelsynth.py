import functools

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    DotProduct,
    ExpSineSquared,
    RBF,
    RationalQuadratic,
    WhiteKernel,
    ConstantKernel,
)

from metaforecast.synth.generators._base import PureSyntheticGenerator

LENGTH = 1024
KERNEL_BANK = [
    ExpSineSquared(periodicity=24 / LENGTH),  # H
    ExpSineSquared(periodicity=48 / LENGTH),  # 0.5H
    ExpSineSquared(periodicity=96 / LENGTH),  # 0.25H
    ExpSineSquared(periodicity=24 * 7 / LENGTH),  # H
    ExpSineSquared(periodicity=48 * 7 / LENGTH),  # 0.5H
    ExpSineSquared(periodicity=96 * 7 / LENGTH),  # 0.25H
    ExpSineSquared(periodicity=7 / LENGTH),  # D
    ExpSineSquared(periodicity=14 / LENGTH),  # 0.5D
    ExpSineSquared(periodicity=30 / LENGTH),  # D
    ExpSineSquared(periodicity=60 / LENGTH),  # 0.5D
    ExpSineSquared(periodicity=365 / LENGTH),  # D
    ExpSineSquared(periodicity=365 * 2 / LENGTH),  # 0.5D
    ExpSineSquared(periodicity=4 / LENGTH),  # W
    ExpSineSquared(periodicity=26 / LENGTH),  # W
    ExpSineSquared(periodicity=52 / LENGTH),  # W
    ExpSineSquared(periodicity=4 / LENGTH),  # M
    ExpSineSquared(periodicity=6 / LENGTH),  # M
    ExpSineSquared(periodicity=12 / LENGTH),  # M
    ExpSineSquared(periodicity=4 / LENGTH),  # Q
    ExpSineSquared(periodicity=4 * 10 / LENGTH),  # Q
    ExpSineSquared(periodicity=10 / LENGTH),  # Q
    DotProduct(sigma_0=0.0),
    DotProduct(sigma_0=1.0),
    DotProduct(sigma_0=10.0),
    RBF(length_scale=0.1),
    RBF(length_scale=1.0),
    RBF(length_scale=10.0),
    RationalQuadratic(alpha=0.1),
    RationalQuadratic(alpha=1.0),
    RationalQuadratic(alpha=10.0),
    WhiteKernel(noise_level=0.1),
    WhiteKernel(noise_level=1.0),
    ConstantKernel(),
]


class KernelSynth(PureSyntheticGenerator):
    """
    # https://github.com/amazon-science/chronos-forecasting/issues/62

    ks = KernelSynth(max_kernels=3, freq='M',n_obs=50)

    ks.transform(n_series=3)
    """

    def __init__(self, max_kernels: int, n_obs: int, freq: str):
        super().__init__(alias='KS')

        self.kernels = KERNEL_BANK
        self.max_kernels = max_kernels
        self.freq = freq
        self.n_obs = n_obs
        self.x = np.linspace(0, 1, self.n_obs)

    def transform(self, n_series: int):
        dt = pd.date_range(start=self.START,
                           periods=self.n_obs,
                           freq=self.freq)

        dataset = []
        for _ in range(n_series):
            ts = self._create_synthetic_ts()

            ts_df = {
                'unique_id': f'KS_UID{self.counter}',
                'ds': dt,
                'y': ts,
            }

            self.counter += 1

            dataset.append(pd.DataFrame(ts_df))

        df = pd.concat(dataset).reset_index(drop=True)

        return df

    def _create_synthetic_ts(self):
        selected_kernels = np.random.choice(
            self.kernels, np.random.randint(1, self.max_kernels + 1), replace=True
        )
        kernel = functools.reduce(self.random_binary_map, selected_kernels)

        ts = self.sample_from_gpr(self.x, kernel)

        return ts

    @staticmethod
    def random_binary_map(a, b):
        binary_maps = [lambda x, y: x + y, lambda x, y: x * y]
        return np.random.choice(binary_maps)(a, b)

    @staticmethod
    def sample_from_gpr(x, kernel):
        gpr = GaussianProcessRegressor(kernel=kernel)

        ts = gpr.sample_y(x[:, None], n_samples=1, random_state=None)
        ts = ts.squeeze()

        return ts
