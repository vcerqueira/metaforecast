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

    def __init__(self, seas_period: int, log: bool = True):
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

        """
        super().__init__(alias='MBB')

        self.log = log
        self.seas_period = seas_period

    def _create_synthetic_ts(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ts = df['y'].copy().values

        synth_ts = _SeasonalMBB.create_bootstrap(
            ts, seas_period=self.seas_period, log=self.log
        )

        df["y"] = synth_ts

        return df
