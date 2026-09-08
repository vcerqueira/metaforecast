"""Bundle-based pure synthetic time series generator.

Implements the four difficulty-conditioned bundle types from Cazaux et al. (2026):
Seasonal-Trend (ST), Non-stationary Regime (NR), Long Memory (LM), and
Volatility Events (VE).  Each bundle mixes its components using Dirichlet-
distributed variance allocation so that the output has approximately unit
variance.
"""

import math
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from metaforecast.synth.generators.base import PureSyntheticGenerator

# Default periods matching common time series frequencies
_DEFAULT_PERIODS = (24, 48, 96, 168, 336)

# Bundle names
BUNDLE_TYPES = ("ST", "NR", "LM", "VE")


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _safe_unit_var(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Standardise *x* to zero mean and unit variance."""
    x = x - x.mean()
    v = x.var() + eps
    return x / np.sqrt(v)


def _trunc_normal(
    rng: np.random.Generator,
    mu: float,
    sigma: float,
    lo: float,
    hi: float,
    max_tries: int = 50,
) -> float:
    for _ in range(max_tries):
        x = rng.normal(mu, sigma)
        if lo <= x <= hi:
            return float(x)
    return float(np.clip(rng.normal(mu, sigma), lo, hi))


def _dirichlet_weights(
    rng: np.random.Generator,
    targets: Sequence[float],
    concentration: float,
) -> np.ndarray:
    t = np.asarray(targets, dtype=np.float64)
    t = np.maximum(t, 1e-6)
    t = t / t.sum()
    alpha = t * float(concentration)
    return rng.dirichlet(alpha)


def _mix_components(
    components: list[np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    """Variance-allocation mixing: y(t) = sum_k sqrt(w_k) * c_k_tilde(t)."""
    y = np.zeros_like(components[0], dtype=np.float64)
    for w, c in zip(weights, components):
        y += np.sqrt(max(1e-12, float(w))) * _safe_unit_var(c.astype(np.float64))
    return y.astype(np.float64)


# ---------------------------------------------------------------------------
# Component generators
# ---------------------------------------------------------------------------

def _gen_fourier(
    rng: np.random.Generator,
    T: int,
    d: float,
    allowed_periods: Sequence[int] = _DEFAULT_PERIODS,
) -> np.ndarray:
    """Fourier seasonal component (Algorithm 1, line 6)."""
    d = _clip01(d)
    t = np.arange(T, dtype=np.float64)

    alpha_decay = 2.0 - 1.0 * d
    K = max(1, int(1 + 5 * d))
    P = int(rng.choice(allowed_periods))

    A0 = float(rng.uniform(0.7, 1.4))
    base_phase = float(rng.uniform(0.0, 2.0 * math.pi))

    sig = np.zeros(T, dtype=np.float64)
    for k in range(1, K + 1):
        ak = A0 / (k**alpha_decay)
        ph = base_phase if k == 1 else float(rng.uniform(0.0, 2.0 * math.pi))
        sig += ak * np.sin(2.0 * math.pi * k * t / P + ph)
    return sig / max(1, K)


def _gen_trend(rng: np.random.Generator, T: int, d: float) -> np.ndarray:
    """Deterministic trend: linear, quadratic, or slow exponential."""
    d = _clip01(d)
    t = np.arange(T, dtype=np.float64)
    u = rng.random()

    if u < 0.50:
        slope = rng.uniform(-0.01 - 0.05 * d, 0.01 + 0.05 * d)
        intercept = rng.uniform(-1.0, 1.0)
        return intercept + slope * t

    if u < 0.80:
        c_scale = 0.002 + 0.018 * d
        c2 = rng.uniform(-c_scale, c_scale)
        c1 = rng.uniform(-0.05 - 0.10 * d, 0.05 + 0.10 * d)
        c0 = rng.uniform(-1.0, 1.0)
        return c2 * t**2 + c1 * t + c0

    a = rng.uniform(0.5, 1.5)
    b = rng.uniform(-0.005 - 0.02 * d, 0.005 + 0.02 * d)
    return a * np.exp(b * t)


def _gen_ar_residual(rng: np.random.Generator, T: int, d: float) -> np.ndarray:
    """AR(1) residual with persistence increasing with difficulty."""
    phi = rng.uniform(0.3 + 0.2 * d, 0.7 + 0.2 * d)
    if rng.random() < 0.15:
        phi = -phi
    sigma = rng.uniform(0.02, 0.08 + 0.20 * d)
    eps = rng.normal(scale=sigma, size=T)
    x = np.zeros(T, dtype=np.float64)
    for i in range(1, T):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def _gen_regime(rng: np.random.Generator, T: int, d: float) -> np.ndarray:
    """Markov-switching AR process (Algorithm 2)."""
    d = _clip01(d)
    M = int(rng.choice([2, 3, 4]))

    p_center = 0.995 - 0.095 * d
    p_band = 0.002 + 0.010 * d
    p_stay = rng.uniform(
        np.clip(p_center - p_band, 0.85, 0.999),
        np.clip(p_center + p_band, 0.85, 0.999),
    )

    means = rng.normal(scale=0.5 + 1.5 * d, size=M)
    slopes = rng.normal(scale=0.02 + 0.08 * d, size=M)
    phi = rng.uniform(0.2, 0.8, size=M)
    eps_scale = 0.03 + 0.17 * d

    state = int(rng.integers(0, M))
    x = np.zeros(T, dtype=np.float64)
    for t in range(1, T):
        if rng.random() > p_stay:
            state = int(rng.integers(0, M))
        x[t] = means[state] + slopes[state] * t + phi[state] * x[t - 1] + rng.normal(scale=eps_scale)
    return x


def _gen_stoch_trend(rng: np.random.Generator, T: int, d: float) -> np.ndarray:
    """Stochastic trend: random walk, GBM, or OU (Algorithm 2, line 6)."""
    d = _clip01(d)
    u = rng.random()

    if u < 0.40:
        step = rng.normal(scale=rng.uniform(0.01, 0.30 + 0.50 * d), size=T)
        return np.cumsum(step)

    if u < 0.80:
        mu = rng.uniform(-0.02, 0.02)
        sigma = rng.uniform(0.05, 0.30 + 0.30 * d)
        W = rng.normal(size=T).cumsum()
        t = np.arange(T, dtype=np.float64)
        return np.exp((mu - 0.5 * sigma**2) * t + sigma * W)

    theta = rng.uniform(0.05, 0.5)
    mu0 = rng.uniform(-1.0, 1.0)
    sigma = rng.uniform(0.05, 0.30 + 0.40 * d)
    x = np.zeros(T, dtype=np.float64)
    for t in range(1, T):
        x[t] = x[t - 1] + theta * (mu0 - x[t - 1]) + sigma * rng.normal()
    return x


def _gen_arfima(rng: np.random.Generator, T: int, d_frac: float) -> np.ndarray:
    """Approximate ARFIMA(0, d, 0) via truncated MA(∞) coefficients."""
    N = T + 200
    w = rng.normal(size=N)
    k = np.arange(N)
    coeff = np.exp(np.log(np.abs(d_frac - k + 1) + 1e-12) - np.log(k + 1))
    series = np.convolve(w, coeff)[:N]
    return series[-T:]


def _gen_long_memory(rng: np.random.Generator, T: int, d: float) -> np.ndarray:
    """Long memory via ARFIMA or approximate fBM (Algorithm 3)."""
    d = _clip01(d)

    if rng.random() < 0.60:
        d_frac = rng.uniform(-0.45, 0.45)
        x = _gen_arfima(rng, T, d_frac)
        x = x + (0.15 + 0.35 * d) * np.cumsum(rng.normal(scale=0.02, size=T))
        return x

    H = rng.uniform(0.6, 0.7 + 0.2 * d)
    phi = 0.6 + 0.35 * (H - 0.6) / 0.3
    z = rng.normal(size=T)
    y = np.zeros(T, dtype=np.float64)
    for t in range(1, T):
        y[t] = phi * y[t - 1] + z[t]
    return y


def _gen_garch(rng: np.random.Generator, T: int, d: float) -> np.ndarray:
    """GARCH(1,1) process (Algorithm 4, lines 1-6)."""
    d = _clip01(d)
    alpha1 = rng.uniform(0.05, 0.20)
    beta1 = rng.uniform(0.60 + 0.20 * d, 0.98)
    beta1 = min(beta1, 0.99 - alpha1 - 1e-3)
    omega = rng.uniform(0.01, 0.05) * (1 - alpha1 - beta1)

    var = np.ones(T) * omega / max(1e-6, 1 - alpha1 - beta1)
    eps = np.zeros(T, dtype=np.float64)
    z = rng.normal(size=T)

    for t in range(1, T):
        var[t] = omega + alpha1 * eps[t - 1] ** 2 + beta1 * var[t - 1]
        var[t] = max(var[t], 1e-12)
        eps[t] = z[t] * np.sqrt(var[t])
    return eps


def _gen_hawkes_spikes(rng: np.random.Generator, T: int, d: float) -> np.ndarray:
    """Hawkes self-exciting spike process (Algorithm 4, lines 7-10)."""
    d = _clip01(d)
    lam0 = rng.uniform(0.001 + 0.01 * d, 0.01 + 0.07 * d)
    spike_amp = rng.uniform(0.5 + 1.5 * d, 2.0 + 5.0 * d)
    alpha_h = rng.uniform(0.10, 0.80)
    beta_h = rng.uniform(0.50, 2.00)

    events = np.zeros(T, dtype=np.float64)
    intensity = lam0
    for t in range(T):
        if rng.random() < min(0.95, intensity):
            events[t] = 1.0
        intensity = lam0 + intensity * np.exp(-beta_h) + alpha_h * events[t]

    kernel = np.exp(-np.arange(10, dtype=np.float64) / 2.0)
    spikes = np.convolve(events, kernel, mode="same") * spike_amp
    return spikes


# ---------------------------------------------------------------------------
# Bundle assemblers
# ---------------------------------------------------------------------------

def _difficulty_profile(d: float) -> str:
    d = _clip01(d)
    if d < 0.30:
        return "easy"
    if d < 0.70:
        return "medium"
    return "hard"


def _noise_sigma(d: float, lo: float = 0.01, hi: float = 0.20) -> float:
    return lo + _clip01(d) * (hi - lo)


def _draw_st(
    rng: np.random.Generator,
    T: int,
    d: float,
    allowed_periods: Sequence[int],
) -> np.ndarray:
    """Seasonal-Trend bundle (Algorithm 1)."""
    prof = _difficulty_profile(d)
    s = _gen_fourier(rng, T, d, allowed_periods)
    tr = _gen_trend(rng, T, d)
    ar = _gen_ar_residual(rng, T, d)

    if prof == "easy":
        targets, conc = [0.92, 0.06, 0.02], 120.0
        sigma_obs = _noise_sigma(d, 0.003, 0.03)
    elif prof == "medium":
        targets, conc = [0.70, 0.18, 0.12], 35.0
        sigma_obs = _noise_sigma(d, 0.01, 0.08)
    else:
        targets, conc = [0.50, 0.20, 0.30], 10.0
        sigma_obs = _noise_sigma(d, 0.03, 0.20)

    w = _dirichlet_weights(rng, targets, conc)
    y = _mix_components([s, tr, ar], w)
    y += rng.normal(scale=sigma_obs, size=T)
    return y


def _draw_nr(
    rng: np.random.Generator,
    T: int,
    d: float,
) -> np.ndarray:
    """Non-stationary Regime bundle (Algorithm 2)."""
    prof = _difficulty_profile(d)
    g = _gen_regime(rng, T, d)
    st = _gen_stoch_trend(rng, T, d)
    ar = _gen_ar_residual(rng, T, d)

    if prof == "easy":
        targets, conc = [0.55, 0.35, 0.10], 80.0
        sigma_obs = _noise_sigma(d, 0.01, 0.05)
    elif prof == "medium":
        targets, conc = [0.50, 0.30, 0.20], 25.0
        sigma_obs = _noise_sigma(d, 0.03, 0.12)
    else:
        targets, conc = [0.45, 0.25, 0.30], 8.0
        sigma_obs = _noise_sigma(d, 0.06, 0.35)

    w = _dirichlet_weights(rng, targets, conc)
    y = _mix_components([g, st, ar], w)
    y += rng.normal(scale=sigma_obs, size=T)
    return y


def _draw_lm(
    rng: np.random.Generator,
    T: int,
    d: float,
    allowed_periods: Sequence[int],
) -> np.ndarray:
    """Long Memory bundle (Algorithm 3)."""
    prof = _difficulty_profile(d)
    lm = _gen_long_memory(rng, T, d)
    comps = [lm]

    p_seas = {"easy": 0.05, "medium": 0.25, "hard": 0.45}[prof]
    if rng.random() < p_seas:
        comps.append(_gen_fourier(rng, T, 0.3 * d, allowed_periods))

    res = rng.normal(scale=_noise_sigma(d, 0.01, 0.25), size=T)
    comps.append(res)

    n_comps = len(comps)
    if prof == "easy":
        targets = [0.90, 0.10] if n_comps == 2 else [0.80, 0.10, 0.10]
        conc, sigma_obs = 90.0, _noise_sigma(d, 0.003, 0.03)
    elif prof == "medium":
        targets = [0.75, 0.25] if n_comps == 2 else [0.70, 0.12, 0.18]
        conc, sigma_obs = 25.0, _noise_sigma(d, 0.01, 0.10)
    else:
        targets = [0.60, 0.40] if n_comps == 2 else [0.55, 0.15, 0.30]
        conc, sigma_obs = 8.0, _noise_sigma(d, 0.03, 0.25)

    w = _dirichlet_weights(rng, targets, conc)
    y = _mix_components(comps, w)
    y += rng.normal(scale=sigma_obs, size=T)
    return y


def _draw_ve(
    rng: np.random.Generator,
    T: int,
    d: float,
) -> np.ndarray:
    """Volatility Events bundle (Algorithm 4)."""
    prof = _difficulty_profile(d)

    ar_phi = rng.uniform(0.2, 0.8)
    ar_sigma = rng.uniform(0.10, 0.40)
    ar_eps = rng.normal(scale=ar_sigma, size=T)
    mean_ar = np.zeros(T, dtype=np.float64)
    for t in range(1, T):
        mean_ar[t] = ar_phi * mean_ar[t - 1] + ar_eps[t]

    garch = _gen_garch(rng, T, d)
    spikes = _gen_hawkes_spikes(rng, T, d)

    if prof == "easy":
        targets, conc = [0.60, 0.30, 0.10], 80.0
        sigma_obs = _noise_sigma(d, 0.005, 0.03)
    elif prof == "medium":
        targets, conc = [0.40, 0.35, 0.25], 25.0
        sigma_obs = _noise_sigma(d, 0.01, 0.08)
    else:
        targets, conc = [0.25, 0.35, 0.40], 8.0
        sigma_obs = _noise_sigma(d, 0.03, 0.20)

    w = _dirichlet_weights(rng, targets, conc)
    y = _mix_components([mean_ar, garch, spikes], w)
    y += rng.normal(scale=sigma_obs, size=T)
    return y


_BUNDLE_DISPATCH = {
    "ST": _draw_st,
    "NR": _draw_nr,
    "LM": _draw_lm,
    "VE": _draw_ve,
}


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class BundleSynth(PureSyntheticGenerator):
    """Generate synthetic time series using difficulty-conditioned bundle archetypes.

    Creates time series from four temporal archetypes — Seasonal-Trend (ST),
    Non-stationary Regime (NR), Long Memory (LM), and Volatility Events (VE) —
    each assembled from standardised components mixed with Dirichlet-distributed
    variance allocation.  A scalar difficulty parameter *d ∈ [0, 1]* controls the
    complexity of every generated series.

    Based on the generator from Cazaux, Ásgeirsson & Stefánsson (2026) [1].

    Parameters
    ----------
    n_obs : int
        Number of time steps per series.
    freq : str
        Pandas frequency string for the generated timestamps (e.g. ``'h'``,
        ``'D'``, ``'ME'``).
    bundle : str or list of str, default ``("ST", "NR", "LM", "VE")``
        Which bundle type(s) to use.  If a list, each series draws a bundle
        uniformly at random (or according to *bundle_weights*).
    bundle_weights : array-like or None
        Probability weights over *bundle* (must sum to 1).  ``None`` means
        uniform.
    difficulty_mode : str, default ``"default"``
        How the difficulty scalar *d* is sampled for each series:

        * ``"default"`` — three-component truncated-normal mixture (Eq. 4).
        * ``"uniform"`` — ``d ~ U(0, 1)``.
        * ``"easy"``    — ``d ~ Beta(2, 5)``, skewed toward 0.
        * ``"medium"``  — ``d ~ Beta(2, 2)``, centred at 0.5.
        * ``"hard"``    — ``d ~ Beta(5, 2)``, skewed toward 1.
    allowed_periods : tuple of int
        Candidate seasonal periods for the ST and LM bundles.
    seed : int or None
        Random seed for reproducibility.  ``None`` means non-deterministic.

    References
    ----------
    [1] Cazaux, H., Ásgeirsson, E. I., & Stefánsson, H. (2026).
    "Does Synthetic Data Help?  Empirical Evidence from Deep Learning
    Time Series Forecasters."  arXiv preprint arXiv:2605.06032.

    Examples
    --------
    >>> from metaforecast.synth import BundleSynth
    >>>
    >>> # Generate 50 Seasonal-Trend series of length 336
    >>> gen = BundleSynth(n_obs=336, freq='h', bundle='ST', seed=42)
    >>> df = gen.transform(n_series=50)
    >>>
    >>> # Generate mixed-bundle series with default difficulty
    >>> gen = BundleSynth(n_obs=96, freq='h', seed=0)
    >>> df = gen.transform(n_series=200)
    """

    def __init__(
        self,
        n_obs: int,
        freq: str,
        bundle: str | Sequence[str] = BUNDLE_TYPES,
        bundle_weights: Optional[Sequence[float]] = None,
        difficulty_mode: str = "default",
        allowed_periods: Sequence[int] = _DEFAULT_PERIODS,
        seed: Optional[int] = None,
    ):
        super().__init__(alias="Bundle")

        self.n_obs = n_obs
        self.freq = freq
        self.allowed_periods = tuple(int(p) for p in allowed_periods)
        self.seed = seed

        if isinstance(bundle, str):
            self.bundles = [bundle.upper()]
        else:
            self.bundles = [b.upper() for b in bundle]
        for b in self.bundles:
            if b not in BUNDLE_TYPES:
                raise ValueError(f"Unknown bundle '{b}'. Expected one of {BUNDLE_TYPES}.")

        if bundle_weights is None:
            self._bundle_weights = np.ones(len(self.bundles)) / len(self.bundles)
        else:
            w = np.asarray(bundle_weights, dtype=np.float64)
            if len(w) != len(self.bundles):
                raise ValueError("`bundle_weights` length must match `bundle`.")
            self._bundle_weights = w / w.sum()

        self.difficulty_mode = difficulty_mode.lower()

    # ------------------------------------------------------------------
    # Difficulty sampling (Eq. 4 and named modes)
    # ------------------------------------------------------------------

    def _sample_difficulty(self, rng: np.random.Generator) -> float:
        mode = self.difficulty_mode
        if mode == "uniform":
            return float(rng.uniform(0, 1))
        if mode == "easy":
            return float(rng.beta(2, 5))
        if mode == "medium":
            return float(rng.beta(2, 2))
        if mode == "hard":
            return float(rng.beta(5, 2))
        # default: three-component truncated-normal mixture
        u = rng.random()
        if u < 0.30:
            return _trunc_normal(rng, 0.20, 0.08, 0.00, 0.35)
        if u < 0.70:
            return _trunc_normal(rng, 0.50, 0.10, 0.35, 0.65)
        return _trunc_normal(rng, 0.80, 0.08, 0.65, 1.00)

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def _create_synthetic_ts(self, rng: np.random.Generator, **kwargs) -> np.ndarray:
        d = self._sample_difficulty(rng)
        bundle_name = rng.choice(self.bundles, p=self._bundle_weights)

        draw_fn = _BUNDLE_DISPATCH[bundle_name]
        if bundle_name in ("ST", "LM"):
            return draw_fn(rng, self.n_obs, d, self.allowed_periods)
        return draw_fn(rng, self.n_obs, d)

    def transform(self, n_series: int, **kwargs) -> pd.DataFrame:
        """Generate *n_series* synthetic time series.

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
                    self.id_col: f"Bundle_UID{self.counter}",
                    self.time_col: dt,
                    self.target_col: ts,
                }
            )
            self.counter += 1
            dataset.append(ts_df)

        return pd.concat(dataset).reset_index(drop=True)
