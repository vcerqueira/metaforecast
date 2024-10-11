from scipy.interpolate import CubicSpline
import pandas as pd
import numpy as np

from metaforecast.synth.generators._base import SemiSyntheticTransformer


class MagnitudeWarping(SemiSyntheticTransformer):
    """ MagnitudeWarping

    Apply magnitude warping to each time series in a dataset

    References:
        Um, T. T., Pfister, F. M., Pichler, D., Endo, S., Lang, M., Hirche, S., ... & Kulić, D. (2017, November).
        Data augmentation of wearable sensor data for parkinson’s disease monitoring using convolutional neural
        networks. In Proceedings of the 19th ACM international conference on multimodal interaction (pp. 216-220).

    Example usage (check notebooks for extended examples):
    >>> import pandas as pd
    >>> from datasetsforecast.m3 import M3
    >>> from neuralforecast import NeuralForecast
    >>> from neuralforecast.models import NHITS
    >>>
    >>> from metaforecast.synth import MagnitudeWarping
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
    >>> tsgen = MagnitudeWarping(sigma=0.2, knot=4)
    >>>
    >>> # Applying time warping to each time series in the dataset
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

    def __init__(self, sigma: float = 0.2, knot=4, rename_uids: bool = True):
        """
        :param sigma: Scaling parameter for the warping factor
        :type sigma: float. Defaults to 0.2

        :param knot: Number of knots for the CubicSpline
        :type knot: int. Defaults to 4

        :param rename_uids: whether to rename the original unique_id's
        :type rename_uids: bool
        """
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
