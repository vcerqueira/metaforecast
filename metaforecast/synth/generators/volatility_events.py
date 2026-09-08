"""Volatility Events (VE) pure synthetic time series generator.

Models financial-style dynamics by combining a mean AR(1) process,
GARCH(1,1) volatility clustering, and Hawkes self-exciting event spikes,
mixed via Dirichlet variance allocation.

Based on Algorithm 4 from Cazaux, Ásgeirsson & Stefánsson (2026) [1]_.

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
    gen_garch,
    gen_hawkes_spikes,
    mix_components,
    noise_sigma,
    sample_difficulty,
)
from metaforecast.synth.generators.base import PureSyntheticGenerator


class VolatilityEvents(PureSyntheticGenerator):
    """Generate synthetic time series with volatility clustering and event spikes.

    Each series is a Dirichlet-weighted mix of three components:

    * **Mean AR(1)** — a baseline autoregressive process.
    * **GARCH(1,1)** — captures time-varying volatility clustering;
      persistence increases with difficulty.
    * **Hawkes spikes** — self-exciting point process whose baseline
      intensity and spike amplitude both scale with difficulty.

    Plus i.i.d. Gaussian observation noise.

    At low difficulty the mean AR dominates; at high difficulty the GARCH and
    Hawkes components contribute most of the variance.

    Parameters
    ----------
    n_obs : int
        Number of time steps per series.
    freq : str
        Pandas frequency string.
    difficulty_mode : str, default ``"default"``
        How the difficulty scalar is sampled — see
        :func:`~metaforecast.synth.generators._bundle_components.sample_difficulty`.
    seed : int or None
        Random seed.

    Examples
    --------
    >>> from metaforecast.synth import VolatilityEvents
    >>> gen = VolatilityEvents(n_obs=336, freq='h', seed=42)
    >>> df = gen.transform(n_series=50)
    """

    def __init__(
        self,
        n_obs: int,
        freq: str,
        difficulty_mode: str = "default",
        seed: int | None = None,
    ):
        super().__init__(alias="VE")

        self.n_obs = n_obs
        self.freq = freq
        self.difficulty_mode = difficulty_mode.lower()
        self.seed = seed

    def _create_synthetic_ts(self, rng: np.random.Generator, **kwargs) -> np.ndarray:
        d = sample_difficulty(rng, self.difficulty_mode)
        prof = difficulty_profile(d)

        # Mean AR(1)
        ar_phi = rng.uniform(0.2, 0.8)
        ar_sigma = rng.uniform(0.10, 0.40)
        ar_eps = rng.normal(scale=ar_sigma, size=self.n_obs)
        mean_ar = np.zeros(self.n_obs, dtype=np.float64)
        for t in range(1, self.n_obs):
            mean_ar[t] = ar_phi * mean_ar[t - 1] + ar_eps[t]

        garch = gen_garch(rng, self.n_obs, d)
        spikes = gen_hawkes_spikes(rng, self.n_obs, d)

        if prof == "easy":
            targets, conc = [0.60, 0.30, 0.10], 80.0
            sigma_obs = noise_sigma(d, 0.005, 0.03)
        elif prof == "medium":
            targets, conc = [0.40, 0.35, 0.25], 25.0
            sigma_obs = noise_sigma(d, 0.01, 0.08)
        else:
            targets, conc = [0.25, 0.35, 0.40], 8.0
            sigma_obs = noise_sigma(d, 0.03, 0.20)

        w = dirichlet_weights(rng, targets, conc)
        y = mix_components([mean_ar, garch, spikes], w)
        y += rng.normal(scale=sigma_obs, size=self.n_obs)
        return y

    def transform(self, n_series: int, **kwargs) -> pd.DataFrame:
        """Generate *n_series* Volatility Events synthetic time series.

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
                    self.id_col: f"VE_UID{self.counter}",
                    self.time_col: dt,
                    self.target_col: ts,
                }
            )
            self.counter += 1
            dataset.append(ts_df)

        return pd.concat(dataset).reset_index(drop=True)
