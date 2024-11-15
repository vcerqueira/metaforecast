import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from metaforecast.synth.generators.base import SemiSyntheticTransformer


class MagnitudeWarping(SemiSyntheticTransformer):
    """Transform time series by applying smooth magnitude variations.

    Implements magnitude warping using random smooth functions to modify
    series amplitudes while preserving temporal patterns.

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
        """Initialize magnitude warping transformer with smoothing parameters.

        Parameters
        ----------
        sigma : float, default=0.2
            Controls intensity of magnitude warping:
            - Higher values create larger variations
            - Lower values produce subtler changes

        knot : int, default=4
            Number of control points for cubic spline warping:
            - More knots: More flexible warping curve
            - Fewer knots: Smoother transformations
            - Recommended range: 3-10
            Controls smoothness of magnitude changes.

        rename_uids : bool, default=True
            Whether to create new identifiers for warped series:
            - True: New ids as f"MWARP_{counter}"
            - False: Preserve original series ids
            Useful for tracking transformations

        """
        super().__init__(alias="MWARP", rename_uids=rename_uids)

        self.sigma = sigma
        self.knot = knot

    def _create_synthetic_ts(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df_ = df.copy()

        warper = self.get_warper(df_.loc[:, "y"].values)

        synth_values = df_['y'].values * warper

        df_.loc[:, 'y'] = synth_values.astype(df_['y'].dtype)

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
