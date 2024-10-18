import numpy as np
import pandas as pd
from statsmodels.tsa.api import STL
from arch.bootstrap import MovingBlockBootstrap

from metaforecast.synth.generators.base import SemiSyntheticTransformer
from metaforecast.utils.log import LogTransformation


class _SeasonalMBB:
    @staticmethod
    def get_mbb(x: pd.Series, w: int, n_samples: int = 1):
        mbb = MovingBlockBootstrap(block_size=w, x=x)

        xt = mbb.bootstrap(n_samples)
        xt = list(xt)
        mbb_series = xt[0][1]['x']

        return mbb_series

    @classmethod
    def create_bootstrap(cls, y: np.ndarray, seas_period: int, log: bool) -> np.ndarray:
        """ create_bootstrap

        Create a bootstrapped version of a time series using moving blocks bootstrap

        :param y: univariate time series
        :type y: np.array

        :param seas_period: Seasonal period (e.g. 12 for monthly time series)
        :type seas_period: int

        :param log: Whether to transform the time series using the logarithm (to stabilize variance)
        :type log: bool
        """

        if log:
            y = LogTransformation.transform(y)

        try:
            stl = STL(y, period=seas_period).fit()

            try:
                synth_res = cls.get_mbb(stl.resid, seas_period)
            except ValueError:
                synth_res = pd.Series(stl.resid).sample(len(stl.resid), replace=True).values

            synth_ts = stl.trend + stl.seasonal + synth_res
        except ValueError:
            synth_ts = y

        if log:
            synth_ts = LogTransformation.inverse_transform(synth_ts)

        return synth_ts


class SeasonalMBB(SemiSyntheticTransformer):
    """ Seasonal Moving Blocks Bootstrap

    Transform the time series in a dataset using bootstrapping

    References:
        Bandara, K., Hewamalage, H., Liu, Y. H., Kang, Y., & Bergmeir, C. (2021). Improving the
        accuracy of global forecasting models using time series data augmentation.
        Pattern Recognition, 120, 108148.

    Example usage (check notebooks for extended examples):

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
        """
        :param seas_period: Seasonal period (e.g. 12 for monthly time series)
        :type seas_period: int

        :param log: Whether to transform the time series using the logarithm (to stabilize variance)
        :type log: bool
        """
        super().__init__(alias='MBB')

        self.log = log
        self.seas_period = seas_period

    def _create_synthetic_ts(self, df: pd.DataFrame) -> pd.DataFrame:
        ts = df['y'].copy().values

        synth_ts = _SeasonalMBB.create_bootstrap(ts, seas_period=self.seas_period, log=self.log)

        df['y'] = synth_ts

        return df
