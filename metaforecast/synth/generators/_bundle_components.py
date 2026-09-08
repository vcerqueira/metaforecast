"""Shared building blocks for difficulty-conditioned bundle generators.

Provides Dirichlet variance allocation, difficulty sampling, and primitive
signal components (Fourier seasonality, AR residuals, trends, regime
processes, long-memory processes, GARCH, Hawkes spikes) used by the four
bundle generators: :class:`SeasonalTrend`, :class:`NonstationaryRegime`,
:class:`LongMemory`, and :class:`VolatilityEvents`.

Based on the generator framework from Cazaux, Ásgeirsson & Stefánsson
(2026), *"Does Synthetic Data Help?  Empirical Evidence from Deep Learning
Time Series Forecasters"*, arXiv:2605.06032.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

# Default periods matching common time series frequencies
DEFAULT_PERIODS: tuple[int, ...] = (24, 48, 96, 168, 336)

# Valid difficulty modes
DIFFICULTY_MODES = ("default", "uniform", "easy", "medium", "hard")


def clip01(x: float) -> float:
    """Clip *x* to [0, 1]."""
    return float(min(1.0, max(0.0, x)))


def safe_unit_var(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Centre and scale *x* to zero mean and unit variance."""
    x = x - x.mean()
    return x / np.sqrt(x.var() + eps)


def trunc_normal(
    rng: np.random.Generator,
    mu: float,
    sigma: float,
    lo: float,
    hi: float,
    max_tries: int = 50,
) -> float:
    """Sample from a truncated normal via rejection."""
    for _ in range(max_tries):
        x = rng.normal(mu, sigma)
        if lo <= x <= hi:
            return float(x)
    return float(np.clip(rng.normal(mu, sigma), lo, hi))


def difficulty_profile(d: float) -> str:
    """Map difficulty scalar to a named profile."""
    d = clip01(d)
    if d < 0.30:
        return "easy"
    if d < 0.70:
        return "medium"
    return "hard"


def noise_sigma(d: float, lo: float = 0.01, hi: float = 0.20) -> float:
    """Observation-noise sigma that scales linearly with difficulty."""
    return lo + clip01(d) * (hi - lo)


def dirichlet_weights(
    rng: np.random.Generator,
    targets: Sequence[float],
    concentration: float,
) -> np.ndarray:
    """Sample Dirichlet weights concentrated around *targets*.

    Parameters
    ----------
    rng : numpy Generator
    targets : sequence of floats summing to ~1
        Desired variance fractions per component.
    concentration : float
        Higher → weights closer to *targets*; lower → more random.

    Returns
    -------
    np.ndarray
        Weight vector summing to 1.
    """
    t = np.asarray(targets, dtype=np.float64)
    t = np.maximum(t, 1e-6)
    t = t / t.sum()
    alpha = t * float(concentration)
    return rng.dirichlet(alpha)


def mix_components(
    components: list[np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    r"""Variance-allocation mixing: :math:`y(t) = \sum_k \sqrt{w_k}\,\tilde c_k(t)`.

    Each component is standardised to unit variance before mixing so that the
    Dirichlet weights directly control the variance share of each component.

    Parameters
    ----------
    components : list of 1-D arrays (all length *T*)
    weights : 1-D array of Dirichlet weights

    Returns
    -------
    np.ndarray
        Mixed signal of length *T*.
    """
    y = np.zeros_like(components[0], dtype=np.float64)
    for w, c in zip(weights, components):
        y += np.sqrt(max(1e-12, float(w))) * safe_unit_var(c.astype(np.float64))
    return y


def sample_difficulty(rng: np.random.Generator, mode: str = "default") -> float:
    """Sample a difficulty scalar *d ∈ [0, 1]*.

    Parameters
    ----------
    rng : numpy Generator
    mode : str
        One of ``"default"``, ``"uniform"``, ``"easy"``, ``"medium"``,
        ``"hard"``.

    Returns
    -------
    float
        Difficulty value in [0, 1].
    """
    mode = mode.lower()
    if mode == "uniform":
        return float(rng.uniform(0, 1))
    if mode == "easy":
        return float(rng.beta(2, 5))
    if mode == "medium":
        return float(rng.beta(2, 2))
    if mode == "hard":
        return float(rng.beta(5, 2))
    # default: three-component truncated-normal mixture (Eq. 4)
    u = rng.random()
    if u < 0.30:
        return trunc_normal(rng, 0.20, 0.08, 0.00, 0.35)
    if u < 0.70:
        return trunc_normal(rng, 0.50, 0.10, 0.35, 0.65)
    return trunc_normal(rng, 0.80, 0.08, 0.65, 1.00)



def gen_fourier(
    rng: np.random.Generator,
    T: int,
    d: float,
    allowed_periods: Sequence[int] = DEFAULT_PERIODS,
) -> np.ndarray:
    """Fourier seasonal component (Algorithm 1, line 5).

    Superposition of *K* harmonics sharing period *P*, with power-law
    amplitude decay whose exponent decreases with difficulty (more
    harmonics contribute at higher difficulty).
    """
    d = clip01(d)
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


def gen_trend(rng: np.random.Generator, T: int, d: float) -> np.ndarray:
    """Deterministic trend: linear, quadratic, or slow exponential."""
    d = clip01(d)
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


def gen_ar_residual(rng: np.random.Generator, T: int, d: float) -> np.ndarray:
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


def gen_regime(rng: np.random.Generator, T: int, d: float) -> np.ndarray:
    """Markov-switching AR process with *M* regimes (Algorithm 2)."""
    d = clip01(d)
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
        x[t] = (
            means[state] + slopes[state] * t + phi[state] * x[t - 1] + rng.normal(scale=eps_scale)
        )
    return x


def gen_stoch_trend(rng: np.random.Generator, T: int, d: float) -> np.ndarray:
    """Stochastic trend: random walk, GBM, or Ornstein-Uhlenbeck."""
    d = clip01(d)
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


def gen_arfima(rng: np.random.Generator, T: int, d_frac: float) -> np.ndarray:
    """Approximate ARFIMA(0, *d_frac*, 0) via truncated MA(∞) coefficients."""
    N = T + 200
    w = rng.normal(size=N)
    k = np.arange(N)
    coeff = np.exp(np.log(np.abs(d_frac - k + 1) + 1e-12) - np.log(k + 1))
    series = np.convolve(w, coeff)[:N]
    return series[-T:]


def gen_long_memory(rng: np.random.Generator, T: int, d: float) -> np.ndarray:
    """Long-memory process via ARFIMA or approximate fBM (Algorithm 3)."""
    d = clip01(d)

    if rng.random() < 0.60:
        d_frac = rng.uniform(-0.45, 0.45)
        x = gen_arfima(rng, T, d_frac)
        x = x + (0.15 + 0.35 * d) * np.cumsum(rng.normal(scale=0.02, size=T))
        return x

    H = rng.uniform(0.6, 0.7 + 0.2 * d)
    phi = 0.6 + 0.35 * (H - 0.6) / 0.3
    z = rng.normal(size=T)
    y = np.zeros(T, dtype=np.float64)
    for t in range(1, T):
        y[t] = phi * y[t - 1] + z[t]
    return y


def gen_garch(rng: np.random.Generator, T: int, d: float) -> np.ndarray:
    """GARCH(1,1) process (Algorithm 4, lines 1-6)."""
    d = clip01(d)
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


def gen_hawkes_spikes(rng: np.random.Generator, T: int, d: float) -> np.ndarray:
    """Hawkes self-exciting spike process (Algorithm 4, lines 7-10)."""
    d = clip01(d)
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
