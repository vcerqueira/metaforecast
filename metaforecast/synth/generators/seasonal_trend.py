"""Seasonal-Trend (ST) pure synthetic time series generator.

Combines Fourier seasonality, a deterministic trend, and an autoregressive
residual, mixed via Dirichlet variance allocation.  A difficulty scalar
*d ∈ [0, 1]* controls the number of harmonics, trend magnitude, AR
persistence, and observation noise.

Based on Algorithm 1 from Cazaux, Ásgeirsson & Stefánsson (2026) [1]_.

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
    gen_ar_residual,
    gen_fourier,
    gen_trend,
    mix_components,
    noise_sigma,
    sample_difficulty,
)
from metaforecast.synth.generators.base import PureSyntheticGenerator


class SeasonalTrend(PureSyntheticGenerator):
    """Generate synthetic time series dominated by seasonality and trend.

    Each series is a Dirichlet-weighted mix of three components:

    * **Fourier seasonality** — superposition of *K ∈ [1, 6]* harmonics with
      shared period *P* and power-law amplitude decay.
    * **Deterministic trend** — linear, quadratic, or slow exponential.
    * **AR(1) residual** — autoregressive noise whose persistence grows with
      difficulty.

    Plus i.i.d. Gaussian observation noise scaled by difficulty.

    The Dirichlet concentration and target variance fractions shift from
    ``[0.92, 0.06, 0.02]`` (easy) to ``[0.50, 0.20, 0.30]`` (hard), so that
    easy series are almost purely seasonal while hard series have substantial
    trend and residual energy.

    Parameters
    ----------
    n_obs : int
        Number of time steps per series.
    freq : str
        Pandas frequency string (e.g. ``'h'``, ``'D'``, ``'ME'``).
    difficulty_mode : str, default ``"default"``
        How the difficulty scalar is sampled per series:

        * ``"default"`` — three-component truncated-normal mixture (Eq. 4).
        * ``"uniform"`` — ``d ~ U(0, 1)``.
        * ``"easy"``    — ``d ~ Beta(2, 5)``.
        * ``"medium"``  — ``d ~ Beta(2, 2)``.
        * ``"hard"``    — ``d ~ Beta(5, 2)``.
    allowed_periods : tuple of int
        Candidate seasonal periods for the Fourier component.
    seed : int or None
        Random seed.  ``None`` → non-deterministic.

    Examples
    --------
    >>> from metaforecast.synth import SeasonalTrend
    >>> gen = SeasonalTrend(n_obs=336, freq='h', seed=42)
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
        super().__init__(alias="ST")

        self.n_obs = n_obs
        self.freq = freq
        self.difficulty_mode = difficulty_mode.lower()
        self.allowed_periods = tuple(int(p) for p in allowed_periods)
        self.seed = seed

    def _create_synthetic_ts(self, rng: np.random.Generator, **kwargs) -> np.ndarray:
        d = sample_difficulty(rng, self.difficulty_mode)
        prof = difficulty_profile(d)

        s = gen_fourier(rng, self.n_obs, d, self.allowed_periods)
        tr = gen_trend(rng, self.n_obs, d)
        ar = gen_ar_residual(rng, self.n_obs, d)

        if prof == "easy":
            targets, conc = [0.92, 0.06, 0.02], 120.0
            sigma_obs = noise_sigma(d, 0.003, 0.03)
        elif prof == "medium":
            targets, conc = [0.70, 0.18, 0.12], 35.0
            sigma_obs = noise_sigma(d, 0.01, 0.08)
        else:
            targets, conc = [0.50, 0.20, 0.30], 10.0
            sigma_obs = noise_sigma(d, 0.03, 0.20)

        w = dirichlet_weights(rng, targets, conc)
        y = mix_components([s, tr, ar], w)
        y += rng.normal(scale=sigma_obs, size=self.n_obs)
        return y

    def transform(self, n_series: int, **kwargs) -> pd.DataFrame:
        """Generate *n_series* Seasonal-Trend synthetic time series.

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
                    self.id_col: f"ST_UID{self.counter}",
                    self.time_col: dt,
                    self.target_col: ts,
                }
            )
            self.counter += 1
            dataset.append(ts_df)

        return pd.concat(dataset).reset_index(drop=True)
