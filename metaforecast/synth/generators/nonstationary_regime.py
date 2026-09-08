"""Non-stationary Regime (NR) pure synthetic time series generator.

Combines a Markov-switching AR process, a stochastic trend, and an
autoregressive residual, mixed via Dirichlet variance allocation.
Difficulty *d ∈ [0, 1]* controls the number of regimes, transition
frequency, shift magnitude, and observation noise.

Based on Algorithm 2 from Cazaux, Ásgeirsson & Stefánsson (2026) [1]_.

References
----------
.. [1] Cazaux, H., Ásgeirsson, E. I., & Stefánsson, H. (2026).
   "Does Synthetic Data Help?  Empirical Evidence from Deep Learning
   Time Series Forecasters."  *arXiv preprint arXiv:2605.06032*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from metaforecast.synth.generators._bundle_components import (
    difficulty_profile,
    dirichlet_weights,
    gen_ar_residual,
    gen_regime,
    gen_stoch_trend,
    mix_components,
    noise_sigma,
    sample_difficulty,
)
from metaforecast.synth.generators.base import PureSyntheticGenerator


class NonstationaryRegime(PureSyntheticGenerator):
    """Generate synthetic time series with structural breaks and regime shifts.

    Each series is a Dirichlet-weighted mix of three components:

    * **Markov-switching AR** — *M ∈ {2, 3, 4}* regimes with
      difficulty-scaled means, slopes, and transition probabilities.
    * **Stochastic trend** — random walk, geometric Brownian motion, or
      Ornstein-Uhlenbeck process.
    * **AR(1) residual** — autoregressive noise.

    Plus i.i.d. Gaussian observation noise.

    Parameters
    ----------
    n_obs : int
        Number of time steps per series.
    freq : str
        Pandas frequency string (e.g. ``'h'``, ``'D'``).
    difficulty_mode : str, default ``"default"``
        How the difficulty scalar is sampled — see
        :func:`~metaforecast.synth.generators._bundle_components.sample_difficulty`.
    seed : int or None
        Random seed.

    Examples
    --------
    >>> from metaforecast.synth import NonstationaryRegime
    >>> gen = NonstationaryRegime(n_obs=336, freq='h', seed=42)
    >>> df = gen.transform(n_series=50)
    """

    def __init__(
        self,
        n_obs: int,
        freq: str,
        difficulty_mode: str = "default",
        seed: int | None = None,
    ):
        super().__init__(alias="NR")

        self.n_obs = n_obs
        self.freq = freq
        self.difficulty_mode = difficulty_mode.lower()
        self.seed = seed

    def _create_synthetic_ts(self, rng: np.random.Generator, **kwargs) -> np.ndarray:
        d = sample_difficulty(rng, self.difficulty_mode)
        prof = difficulty_profile(d)

        g = gen_regime(rng, self.n_obs, d)
        st = gen_stoch_trend(rng, self.n_obs, d)
        ar = gen_ar_residual(rng, self.n_obs, d)

        if prof == "easy":
            targets, conc = [0.55, 0.35, 0.10], 80.0
            sigma_obs = noise_sigma(d, 0.01, 0.05)
        elif prof == "medium":
            targets, conc = [0.50, 0.30, 0.20], 25.0
            sigma_obs = noise_sigma(d, 0.03, 0.12)
        else:
            targets, conc = [0.45, 0.25, 0.30], 8.0
            sigma_obs = noise_sigma(d, 0.06, 0.35)

        w = dirichlet_weights(rng, targets, conc)
        y = mix_components([g, st, ar], w)
        y += rng.normal(scale=sigma_obs, size=self.n_obs)
        return y

    def transform(self, n_series: int, **kwargs) -> pd.DataFrame:
        """Generate *n_series* Non-stationary Regime synthetic time series.

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
                    self.id_col: f"NR_UID{self.counter}",
                    self.time_col: dt,
                    self.target_col: ts,
                }
            )
            self.counter += 1
            dataset.append(ts_df)

        return pd.concat(dataset).reset_index(drop=True)
