"""Dominant Shuffle augmentation for time series.

Shuffles the top-*k* dominant frequency components (by magnitude) of a time
series.  Unlike full-spectrum perturbations, limiting changes to dominant
frequencies preserves the main periodicity and trends while generating
meaningful variations.  Shuffling rearranges existing components without
introducing external noise, keeping augmented data close to the original
distribution.

Based on Zhao et al. (2024) [1]_.

References
----------
.. [1] Zhao, K., He, Z., Hung, A., & Zeng, D. (2024). "Dominant Shuffle:
   A Simple Yet Powerful Data Augmentation for Time-series Prediction."
   *arXiv preprint arXiv:2405.16456*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from metaforecast.synth.generators.base import SemiSyntheticTransformer


class DominantShuffle(SemiSyntheticTransformer):
    """Augment time series by shuffling dominant frequency components.

    Applies a discrete Fourier transform to the series, identifies the *k*
    frequency bins with highest magnitude, randomly permutes those complex
    coefficients, and transforms back to the time domain.

    This produces augmented series that retain the same spectral energy but
    redistribute it among the dominant modes, creating plausible variations
    without introducing out-of-distribution noise.

    Parameters
    ----------
    k : int, default 3
        Number of dominant frequency components to shuffle.  Higher values
        create more variation; lower values produce subtler changes.  The
        paper recommends tuning *k* per dataset on a validation set.
    rename_uids : bool, default True
        Whether to create new identifiers for augmented series.

    Examples
    --------
    >>> from metaforecast.synth import DominantShuffle
    >>> aug = DominantShuffle(k=3)
    >>> augmented_df = aug.transform(train_df)
    """

    def __init__(self, k: int = 3, rename_uids: bool = True):
        super().__init__(alias="DSHUFFLE", rename_uids=rename_uids)

        self.k = k

    def _create_synthetic_ts(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df_ = df.copy()
        y = df_[self.target_col].values.astype(np.float64)

        F = np.fft.rfft(y)

        n_freqs = len(F)
        k = min(self.k, n_freqs)

        magnitudes = np.abs(F)
        dominant_idx = np.argsort(magnitudes)[-k:]

        dominant_coeffs = F[dominant_idx].copy()
        np.random.shuffle(dominant_coeffs)
        F[dominant_idx] = dominant_coeffs

        augmented = np.fft.irfft(F, n=len(y))

        df_.loc[:, self.target_col] = augmented.astype(df_[self.target_col].dtype)

        return df_
