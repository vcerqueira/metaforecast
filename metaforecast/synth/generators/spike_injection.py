"""Spike Injection augmentation for time series.

Injects structured, periodic spike signals into the time series.  A periodic
pattern is sampled and tiled across the time axis.  Each spike label in the
pattern is mapped to a kernel (tophat, RBF, or linear) with randomised width
and amplitude, and the sum of all kernel evaluations is added to the original
signal.

Based on Algorithm 3 from Auer et al. (2025) [1]_.

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

# Spike patterns and their sampling probabilities (Table 3)
_PATTERNS: list[tuple[list[list[int]], float]] = [
    # Simple (p=0.75)
    ([[0], [0, 1]], 0.75),
    # 3-periodic (p=0.10)
    ([[0, 1, 2], [0, 0, 1]], 0.10),
    # 4-periodic (p=0.10)
    ([[0, 0, 1, 1], [0, 1, 0, 2]], 0.10),
    # Weekly-like (p=0.05)
    ([[0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 1, 2]], 0.05),
]

# Kernel types (Table 2)
_KERNEL_TYPES = ("tophat", "rbf", "linear")


def _tophat_kernel(t: np.ndarray, centre: float, width: float, height: float) -> np.ndarray:
    """Rectangular pulse centred at *centre* with half-width *width*."""
    return np.where(np.abs(t - centre) <= width / 2.0, height, 0.0)


def _rbf_kernel(t: np.ndarray, centre: float, sigma: float, height: float) -> np.ndarray:
    """Gaussian (RBF) bump centred at *centre*."""
    return height * np.exp(-0.5 * ((t - centre) / max(sigma, 1e-6)) ** 2)


def _linear_kernel(t: np.ndarray, centre: float, width: float, height: float) -> np.ndarray:
    """Triangular (linear) spike centred at *centre*."""
    half_w = width / 2.0
    dist = np.abs(t - centre)
    return np.where(dist <= half_w, height * (1.0 - dist / max(half_w, 1e-6)), 0.0)


class SpikeInjection(SemiSyntheticTransformer):
    """Augment time series by injecting structured periodic spikes.

    A periodicity is sampled, then a categorical spike pattern is tiled along
    the time axis.  For each unique spike label a kernel type (tophat, RBF, or
    linear) is chosen with random width and amplitude parameters, and the
    additive spike signal is the sum of all kernel evaluations.

    This augmentation improves generalisation to sharp, transient events by
    exposing the model to a diversity of spike structures.

    Parameters
    ----------
    scale_to_signal : bool, default True
        If ``True``, the spike amplitude range ``[0.5, 3]`` is multiplied by
        the standard deviation of the input series so that spikes are
        proportional to the signal scale.
    rename_uids : bool, default True
        Whether to create new identifiers for augmented series.

    Examples
    --------
    >>> from metaforecast.synth import SpikeInjection
    >>> aug = SpikeInjection()
    >>> augmented_df = aug.transform(train_df)
    """

    def __init__(
        self,
        scale_to_signal: bool = True,
        rename_uids: bool = True,
    ):
        super().__init__(alias="SPIKE", rename_uids=rename_uids)

        self.scale_to_signal = scale_to_signal

    def _create_synthetic_ts(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df_ = df.copy()
        y = df_[self.target_col].values.astype(np.float64)
        T = len(y)

        omega = int(np.random.uniform(10, min(512, T)))

        pattern = self._sample_pattern()
        shift = np.random.randint(0, len(pattern))
        pattern = pattern[shift:] + pattern[:shift]

        s = int(np.random.uniform(max(0, T - omega), T))

        spike_positions = self._tile_pattern(pattern, omega, T, s)

        kernel_type = np.random.choice(_KERNEL_TYPES)

        t_axis = np.arange(T, dtype=np.float64)

        amp_scale = float(np.std(y) + 1e-8) if self.scale_to_signal else 1.0

        unique_labels = set(spike_positions)
        unique_labels.discard(-1)

        label_params: dict[int, tuple[str, float, float]] = {}
        for label in unique_labels:
            width = np.random.uniform(0.05 * omega, 0.20 * omega)
            height = np.random.uniform(0.5, 3.0) * amp_scale
            label_params[label] = (kernel_type, width, height)

        spike_signal = np.zeros(T, dtype=np.float64)
        for t_idx in range(T):
            label = spike_positions[t_idx]
            if label < 0:
                continue
            ktype, width, height = label_params[label]
            if ktype == "tophat":
                spike_signal += _tophat_kernel(t_axis, float(t_idx), width, height)
            elif ktype == "rbf":
                spike_signal += _rbf_kernel(t_axis, float(t_idx), width, height)
            else:
                spike_signal += _linear_kernel(t_axis, float(t_idx), width, height)

        df_.loc[:, self.target_col] = (y + spike_signal).astype(df_[self.target_col].dtype)

        return df_

    @staticmethod
    def _sample_pattern() -> list[int]:
        """Sample a spike pattern according to Table 3 probabilities."""
        u = np.random.random()
        cumulative = 0.0
        for patterns, prob in _PATTERNS:
            cumulative += prob
            if u < cumulative:
                chosen = patterns[np.random.randint(0, len(patterns))]
                return list(chosen)
        return [0]

    @staticmethod
    def _tile_pattern(pattern: list[int], omega: int, T: int, last_spike_pos: int) -> list[int]:
        """Tile *pattern* with spacing *omega*, aligning last spike at *last_spike_pos*.

        Returns a list of length *T* where each element is the spike label
        at that timestep, or -1 for no spike.
        """
        pat_len = len(pattern)
        positions = [-1] * T

        full_period = omega * pat_len

        start = last_spike_pos - full_period
        while start + full_period > 0:
            start -= full_period

        t = start
        while t < T:
            for i, label in enumerate(pattern):
                pos = t + i * omega
                if 0 <= pos < T:
                    positions[pos] = label
            t += full_period

        return positions
