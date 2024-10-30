import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from metaforecast.synth.generators.base import SemiSyntheticTransformer


class TimeWarping(SemiSyntheticTransformer):
    """Transform time series by applying non-linear temporal distortions.

    Implements time warping augmentation by distorting the time axis while
    preserving value relationships.

    References
    ----------
    Um, T. T., et al. (2017).  "Data augmentation of wearable sensor data for parkinson's disease
    monitoring using convolutional neural networks."
    In Proceedings of the 19th ACM International Conference
    on Multimodal Interaction (pp. 216-220).

    Examples
    --------
    >>> import pandas as pd
    >>> from datasetsforecast.m3 import M3
    >>> from neuralforecast import NeuralForecast
    >>> from neuralforecast.models import NHITS
    >>>
    >>> from metaforecast.synth import TimeWarping
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
    >>> tsgen = TimeWarping(sigma=0.2, knot=4)
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
        """Initialize time warping transformer with distortion parameters.

        Parameters
        ----------
        sigma : float, default=0.2
            Controls intensity of temporal distortion:
            - Higher values create larger time shifts
            - Lower values produce subtler variations

        knot : int, default=4
            Number of control points for cubic spline warping:
            - More knots: More complex temporal distortions
            - Fewer knots: Smoother time transformations
            - Recommended range: 3-10
            Controls smoothness of time warping.

        rename_uids : bool, default=True
            Whether to create new identifiers for warped series:
            - True: New ids as f"TWARP_{counter}"
            - False: Preserve original series ids

        """
        super().__init__(alias="TWARP", rename_uids=rename_uids)

        self.sigma = sigma
        self.knot = knot

    def _create_synthetic_ts(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df_ = df.copy()

        df_.loc[:, "y"] = self.apply_time_warping(df_.loc[:, "y"].values)

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
