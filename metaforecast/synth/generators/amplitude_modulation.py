"""Amplitude Modulation augmentation for time series.

Introduces scale trends and changepoints by multiplying the signal with a
piecewise-linear trend.  Changepoints are sampled uniformly, amplitudes at
each segment boundary are drawn from N(1, 1), and the trend is obtained by
linear interpolation between them.

Based on Algorithm 1 from Auer et al. (2025) [1]_.

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


class AmplitudeModulation(SemiSyntheticTransformer):
    """Augment time series by multiplying with a piecewise-linear trend.

    Random changepoints divide the series into segments.  At each boundary
    (including the start and end) an amplitude is drawn from N(1, 1), and the
    modulation trend is obtained by linear interpolation.  The augmented
    series is the element-wise product of the original signal and this trend.

    This augmentation exposes models to non-stationary scale shifts and
    gradual amplitude changes, improving robustness to level changes and
    structural breaks in real-world data.

    Parameters
    ----------
    max_changepoints : int, default 5
        Maximum number of changepoints.  The actual count per series is
        drawn from ``Uniform(0, max_changepoints)``.
    rename_uids : bool, default True
        Whether to create new identifiers for augmented series.

    Examples
    --------
    >>> from metaforecast.synth import AmplitudeModulation
    >>> aug = AmplitudeModulation(max_changepoints=5)
    >>> augmented_df = aug.transform(train_df)
    """

    def __init__(
        self,
        max_changepoints: int = 5,
        rename_uids: bool = True,
    ):
        super().__init__(alias="AMPMOD", rename_uids=rename_uids)

        self.max_changepoints = max_changepoints

    def _create_synthetic_ts(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df_ = df.copy()
        y = df_[self.target_col].values.astype(np.float64)
        T = len(y)

        k = np.random.randint(0, self.max_changepoints + 1)

        if k > 0:
            cps = np.sort(np.random.choice(np.arange(1, T), size=min(k, T - 1), replace=False))
        else:
            cps = np.array([], dtype=int)

        positions = np.concatenate([[0], cps, [T - 1]])
        amplitudes = np.random.normal(loc=1.0, scale=1.0, size=len(positions))

        trend = np.interp(np.arange(T), positions, amplitudes)

        df_.loc[:, self.target_col] = (y * trend).astype(df_[self.target_col].dtype)

        return df_
