from typing import Optional

import numpy as np
import pandas as pd
from arch.bootstrap import MovingBlockBootstrap
from statsmodels.tsa.api import STL

from metaforecast.synth.generators.base import SemiSyntheticTransformer
from metaforecast.utils.log import LogTransformation


class _SeasonalMBB:
    @staticmethod
    def get_mbb(x: pd.Series, w: int, n_samples: int = 1):
        mbb = MovingBlockBootstrap(block_size=w, x=x)

        xt = mbb.bootstrap(n_samples)
        xt = list(xt)
        mbb_series = xt[0][1]["x"]

        return mbb_series

    @classmethod
    def create_bootstrap(cls, y: np.ndarray, seas_period: int, log: bool) -> np.ndarray:
        """Create bootstrapped time series using moving blocks bootstrap.

        Generates a synthetic version of the input series by resampling
        blocks of observations while preserving temporal dependencies
        and seasonal patterns.

        Parameters
        ----------
        y : np.ndarray
            Input time series values.
            Shape: (n_observations,)
            Must be 1-dimensional.

        seas_period : int
            Seasonal period of the time series:
            - 12 for monthly data
            - 4 for quarterly data
            - 7 for daily data with weekly patterns
            Used to determine block size and preserve seasonality.

        log : bool, default=False
            Whether to apply log transformation before bootstrapping:
        """

        if log:
            y = LogTransformation.transform(y)

        try:
            stl = STL(y, period=seas_period).fit()

            try:
                synth_res = cls.get_mbb(stl.resid, seas_period)
            except ValueError:
                synth_res = (
                    pd.Series(stl.resid).sample(len(stl.resid), replace=True).values
                )

            synth_ts = stl.trend + stl.seasonal + synth_res
        except ValueError:
            synth_ts = y

        if log:
            synth_ts = LogTransformation.inverse_transform(synth_ts)

        return synth_ts


class SeasonalMBB(SemiSyntheticTransformer):
    """Transform time series using seasonal moving blocks bootstrap.

    todo add seed

    Creates synthetic variations of time series by resampling blocks
    of observations while preserving seasonal patterns and temporal
    dependencies. Method described in Bandara et al. shown to
    improve forecasting accuracy through augmentation.

    References
    ----------
    Bandara, K., Hewamalage, H., Liu, Y. H., Kang, Y.,
    & Bergmeir, C. (2021). "Improving the accuracy of global
    forecasting models using time series data augmentation."
    Pattern Recognition, 120, 108148.

    Examples
    --------
    >>> import pandas as pd
    >>> from datasetsforecast.m3 import M3
    >>> from neuralforecast import NeuralForecast
    >>> from neuralforecast.models import NHITS
    >>>
    >>> from metaforecast.synth import SeasonalMBB
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
    >>> tsgen = SeasonalMBB(seas_period=12)
    >>>
    >>> # Creating time series using kernelsynth
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
                 seas_period: int,
                 log: bool = True,
                 max_samples_in_stl: Optional[int] = None):
        """Initialize seasonal moving blocks bootstrap transformer.

        Parameters
        ----------
        seas_period : int
            Seasonal period of the time series:
            - 12 for monthly data
            - 4 for quarterly data
            - etc

        log : bool, default=True
            Whether to apply log transformation before bootstrapping:

        max_samples_in_stl: int, Optional, default=None
            Whether to break down the MBB process into different chunks with size max_samples_in_stl.
            This can be useful for large time series where STL apparently struggles
            If None (default behaviour), chunking is not done.

        """
        super().__init__(alias='MBB')

        self.log = log
        self.seas_period = seas_period
        self.max_samples_in_stl = max_samples_in_stl

    def _create_synthetic_ts(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:

        if self.max_samples_in_stl is not None:
            df_chunks = self._chunk_dataframe(df, self.max_samples_in_stl)

            mbb_chunks = []
            for df_chk in df_chunks:
                ts_ = df_chk[self.target_col].copy().values

                synth_tsc = _SeasonalMBB.create_bootstrap(
                    ts_, self.seas_period, log=self.log
                )

                df_chk[self.target_col] = synth_tsc

                mbb_chunks.append(df_chk)

            mbb_df = pd.concat(mbb_chunks)
            mbb_df.index = df.index

            df = mbb_df.copy()
        else:
            ts = df[self.target_col].copy().values

            synth_ts = _SeasonalMBB.create_bootstrap(
                ts, seas_period=self.seas_period, log=self.log
            )

            df[self.target_col] = synth_ts

        return df

    @staticmethod
    def _chunk_dataframe(df: pd.DataFrame, chunk_size: int):
        total_rows = df.shape[0]

        if total_rows <= chunk_size:
            return [df]

        n_full_chunks = total_rows // chunk_size
        last_chunk_size = total_rows % chunk_size

        if last_chunk_size > 0:
            chunks = [df.iloc[i * chunk_size:(i + 1) * chunk_size] for i in range(n_full_chunks - 1)]
            chunks.append(df.iloc[(n_full_chunks - 1) * chunk_size:])
        else:
            chunks = [df.iloc[i * chunk_size:(i + 1) * chunk_size] for i in range(n_full_chunks)]

        return chunks
