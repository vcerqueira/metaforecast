import numpy as np
import pandas as pd

from metaforecast.synth.generators.base import SemiSyntheticTransformer


class Scaling(SemiSyntheticTransformer):
    """Transform time series by applying controlled scaling operations.

    Implements magnitude scaling transformations while preserving temporal
    patterns and relationships.

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
    >>> from metaforecast.synth import Scaling
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
    >>> tsgen = Scaling()
    >>>
    >>> # Scaling each time series in the dataset
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

    def __init__(self, sigma: float = 0.1, rename_uids: bool = True):
        """Initialize scaling transformer with magnitude parameters.

        Parameters
        ----------
        sigma : float, default=0.1
            Standard deviation for random scaling factors:
            - Higher values create larger magnitude changes
            - Lower values produce subtler variations
            Must be positive.

        rename_uids : bool, default=True
            Whether to create new identifiers for scaled series:
            - True: New ids as f"SCALE_{counter}"
            - False: Preserve original series ids
            Useful for tracking transformations

        """
        super().__init__(alias='SCALE', rename_uids=rename_uids)

        self.sigma = sigma

    def _create_synthetic_ts(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df_ = df.copy()

        factor = np.random.normal(loc=1.0, scale=self.sigma, size=df_.shape[0])

        synth_values = df_['y'].values * factor

        df_.loc[:, 'y'] = synth_values.astype(df_['y'].dtype)

        return df_
