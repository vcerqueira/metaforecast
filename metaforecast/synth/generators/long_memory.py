"""Long Memory (LM) pure synthetic time series generator.

Generates series with slowly decaying autocorrelations via ARFIMA(0, d, 0)
or approximate fractional Brownian motion, with an optional seasonal overlay
whose probability increases with difficulty.

Based on Algorithm 3 from Cazaux, Ásgeirsson & Stefánsson (2026) [1]_.

References
----------
.. [1] Cazaux, H., Ásgeirsson, E. I., & Stefánsson, H. (2026).
   "Does Synthetic Data Help?  Empirical Evidence from Deep Learning
   Time Series Forecasters."  *arXiv preprint arXiv:2605.06032*.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from metaforecast.synth.generators._bundle_components import (
    DEFAULT_PERIODS,
    difficulty_profile,
    dirichlet_weights,
    gen_fourier,
    gen_long_memory,
    mix_components,
    noise_sigma,
    sample_difficulty,
)
from metaforecast.synth.generators.base import PureSyntheticGenerator


class LongMemory(PureSyntheticGenerator):
    """Generate synthetic time series with long-range dependence.

    Each series is a Dirichlet-weighted mix of two or three components:

    * **Long-memory process** — ARFIMA(0, *d_frac*, 0) with
      *d_frac* in (-0.45, 0.45), or approximate fractional Brownian motion
      with Hurst exponent *H ∈ [0.6, 0.9]*.
    * **Seasonal overlay** (optional) — Fourier seasonality at reduced
      difficulty; inclusion probability grows with *d*
      (5 % easy → 25 % medium → 45 % hard).
    * **Noise component** — Gaussian residual.

    Plus i.i.d. observation noise.

    Parameters
    ----------
    n_obs : int
        Number of time steps per series.
    freq : str
        Pandas frequency string.
    difficulty_mode : str, default ``"default"``
        How the difficulty scalar is sampled — see
        :func:`~metaforecast.synth.generators._bundle_components.sample_difficulty`.
    allowed_periods : tuple of int
        Candidate seasonal periods for the optional Fourier overlay.
    seed : int or None
        Random seed.

    Examples
    --------
    >>> from metaforecast.synth import LongMemory
    >>> gen = LongMemory(n_obs=336, freq='h', seed=42)
    >>> df = gen.transform(n_series=50)
    """

    def __init__(
        self,
        n_obs: int,
        freq: str,
        difficulty_mode: str = "default",
        allowed_periods: Sequence[int] = DEFAULT_PERIODS,
        seed: int | None = None,
    ):
        super().__init__(alias="LM")

        self.n_obs = n_obs
        self.freq = freq
        self.difficulty_mode = difficulty_mode.lower()
        self.allowed_periods = tuple(int(p) for p in allowed_periods)
        self.seed = seed

    def _create_synthetic_ts(self, rng: np.random.Generator, **kwargs) -> np.ndarray:
        d = sample_difficulty(rng, self.difficulty_mode)
        prof = difficulty_profile(d)

        lm = gen_long_memory(rng, self.n_obs, d)
        comps: list[np.ndarray] = [lm]

        p_seas = {"easy": 0.05, "medium": 0.25, "hard": 0.45}[prof]
        if rng.random() < p_seas:
            comps.append(gen_fourier(rng, self.n_obs, 0.3 * d, self.allowed_periods))

        res = rng.normal(scale=noise_sigma(d, 0.01, 0.25), size=self.n_obs)
        comps.append(res)

        n_comps = len(comps)
        if prof == "easy":
            targets = [0.90, 0.10] if n_comps == 2 else [0.80, 0.10, 0.10]
            conc, sigma_obs = 90.0, noise_sigma(d, 0.003, 0.03)
        elif prof == "medium":
            targets = [0.75, 0.25] if n_comps == 2 else [0.70, 0.12, 0.18]
            conc, sigma_obs = 25.0, noise_sigma(d, 0.01, 0.10)
        else:
            targets = [0.60, 0.40] if n_comps == 2 else [0.55, 0.15, 0.30]
            conc, sigma_obs = 8.0, noise_sigma(d, 0.03, 0.25)

        w = dirichlet_weights(rng, targets, conc)
        y = mix_components(comps, w)
        y += rng.normal(scale=sigma_obs, size=self.n_obs)
        return y

    def transform(self, n_series: int, **kwargs) -> pd.DataFrame:
        """Generate *n_series* Long Memory synthetic time series.

        Parameters
        ----------
        n_series : int
            Number of series to generate.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with columns ``unique_id``, ``ds``, ``y``.
        """
        base_rng = np.random.default_rng(self.seed)
        dt = pd.date_range(start=self.START, periods=self.n_obs, freq=self.freq)

        dataset = []
        for _ in range(n_series):
            child_seed = int(base_rng.integers(0, 2**31))
            rng = np.random.default_rng(child_seed)
            ts = self._create_synthetic_ts(rng)

            ts_df = pd.DataFrame(
                {
                    self.id_col: f"LM_UID{self.counter}",
                    self.time_col: dt,
                    self.target_col: ts,
                }
            )
            self.counter += 1
            dataset.append(ts_df)

        return pd.concat(dataset).reset_index(drop=True)
