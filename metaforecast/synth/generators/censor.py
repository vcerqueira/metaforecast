"""Censor Augmentation for time series.

Clips (censors) the signal from above or below at a randomly sampled
quantile threshold.  This teaches models to handle clipped or saturated
sensor readings, truncated distributions, and hard limits common in
real-world data.

Based on Algorithm 2 from Auer et al. (2025) [1]_.

References
----------
.. [1] Auer, A., Bock, S., Podest, P., Klambauer, G., Klotz, D., &
   Hochreiter, S. (2025). "TiRex: Zero-Shot Forecasting Across Long and
   Short Horizons with Enhanced In-Context Learning."
   *arXiv preprint arXiv:2505.23719*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from metaforecast.synth.generators.base import SemiSyntheticTransformer


class CensorAugmentation(SemiSyntheticTransformer):
    """Augment time series by censoring (clipping) values at a random quantile.

    A quantile level *q* is drawn from U(0, 1) and the empirical quantile of
    the series is computed.  With equal probability the series is clipped from
    below (``max(y, threshold)``) or from above (``min(y, threshold)``).

    Parameters
    ----------
    rename_uids : bool, default True
        Whether to create new identifiers for censored series.

    Examples
    --------
    >>> from metaforecast.synth import CensorAugmentation
    >>> aug = CensorAugmentation()
    >>> augmented_df = aug.transform(train_df)
    """

    def __init__(self, rename_uids: bool = True):
        super().__init__(alias="CENSOR", rename_uids=rename_uids)

    def _create_synthetic_ts(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df_ = df.copy()
        y = df_[self.target_col].values.astype(np.float64)

        q = np.random.uniform(0.0, 1.0)
        threshold = float(np.quantile(y, q))

        bottom = np.random.random() < 0.5

        augmented = np.maximum(y, threshold) if bottom else np.minimum(y, threshold)

        df_.loc[:, self.target_col] = augmented.astype(df_[self.target_col].dtype)

        return df_
