import numpy as np
import pandas as pd

from metaforecast.synth.generators.base import SemiSyntheticTransformer


class Jittering(SemiSyntheticTransformer):
    """Add controlled Gaussian noise to time series for data augmentation.

    Implements time series jittering by adding random Gaussian noise
    to original values.

    References
    ----------
    Um, T. T., et al. (2017). "Data augmentation of wearable sensor data for parkinson's disease
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
    >>> from metaforecast.synth import Jittering
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
    >>> tsgen = Jittering(sigma=0.5)
    >>>
    >>> # Add jittering to each time series
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

    def __init__(self, sigma: float = 0.03, rename_uids: bool = True):
        """Initialize Jittering transformer with noise parameters.

        Parameters
        ----------
        sigma : float, default=0.03
            Standard deviation of Gaussian noise as proportion of series scale:
            - Higher values create more variation
            - Lower values produce subtler changes
            - 0.03 means 3% of series scale
            Must be positive.

        rename_uids : bool, default=True
            Whether to create new identifiers for jittered series:
            - True: New ids as f"JITTER_{counter}"
            - False: Preserve original series ids
            Useful for tracking transformations

        """
        super().__init__(alias="JITTER", rename_uids=rename_uids)

        self.sigma = sigma

    def _create_synthetic_ts(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df_ = df.copy()

        sig = self.sigma * df_["y"].std()

        jitter_ = np.random.normal(loc=0.0, scale=sig, size=df_.shape[0])

        synth_values = df_['y'].values + jitter_

        df_.loc[:, 'y'] = synth_values.astype(df_['y'].dtype)

        return df_
