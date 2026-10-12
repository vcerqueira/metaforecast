"""StatsForecast model backed by a meta-fitted MetaARIMA.

``forecast`` receives one series and returns an ``h``-step mean. The
wrapper rebuilds a one-series frame, then MetaARIMA extracts tsfeatures,
shortlists configurations, and selects by AICc. Coefficient training
stays on :class:`~metaforecast.coseal.metaarima.meta_arima.MetaARIMA`.

Prediction intervals follow ``statsforecast`` ARIMA: native intervals
when ``level`` is set, or conformal intervals when ``prediction_intervals``
is set on this model or on :class:`statsforecast.StatsForecast`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsforecast.models import _TS

from metaforecast.coseal.metaarima.meta_arima import MetaARIMA


class MetaARIMAModel(_TS):
    """Apply a meta-fitted :class:`~metaforecast.coseal.metaarima.meta_arima.MetaARIMA`.

    StatsForecast calls :meth:`forecast` with the series values and the
    horizon. It does not pass ``unique_id``, so this model rebuilds a
    one-series frame from ``y`` using ``freq``. ``fitted=True`` is not
    supported.

    ``StatsForecast.fit`` copies this model per series with :meth:`new`,
    which shares the underlying :class:`MetaARIMA`. Each copy stores its
    own selected ARIMA. ``MetaARIMA.selected_config`` still reflects the
    series fitted last.

    Parameters
    ----------
    metaarima : MetaARIMA
        A meta-fitted model (``MetaARIMA.load`` or ``meta_fit``).
    alias : str, optional
        Name used as the StatsForecast column. Defaults to ``"MetaARIMA"``.
    freq : str, optional
        Inference frequency. Defaults to ``metaarima.freq``.
    season_length : int, optional
        Inference seasonal period. Defaults to ``metaarima.season_length``.
        A value different from meta-training rewrites ``[m]`` in the
        configuration shortlist.

    Examples
    --------
    >>> from statsforecast import StatsForecast
    >>> from statsforecast.models import AutoARIMA, AutoETS
    >>> from metaforecast.coseal import MetaARIMA, MetaARIMAModel
    >>> meta = MetaARIMA.load(freq="ME")
    >>> sf = StatsForecast(
    ...     models=[
    ...         MetaARIMAModel(meta),
    ...         AutoARIMA(season_length=12),
    ...         AutoETS(season_length=12),
    ...     ],
    ...     freq="ME",
    ... )
    >>> sf.fit(train)
    >>> fcst = sf.predict(h=12)
    """

    uses_exog = False

    def __init__(
        self,
        metaarima: MetaARIMA,
        alias: str | None = None,
        freq: str | None = None,
        season_length: int | None = None,
    ):
        if not isinstance(metaarima, MetaARIMA):
            raise TypeError("metaarima must be a MetaARIMA")
        if not metaarima.is_fit:
            raise ValueError("MetaARIMA must be meta-fitted before wrapping")

        self.metaarima = metaarima
        self.alias = alias or "MetaARIMA"
        self.freq = metaarima.freq if freq is None else freq
        self.season_length = metaarima.season_length if season_length is None else season_length
        self.prediction_intervals = None

    def forecast(self, y, h, X=None, X_future=None, level=None, fitted: bool = False):
        """``h``-step forecast of one series.

        ``X`` and ``X_future`` are accepted so the signature matches
        StatsForecast, and ignored. When ``level`` is set and
        ``prediction_intervals`` is unset, intervals come from the
        selected ARIMA. Otherwise they are conformal.
        """
        if fitted:
            raise ValueError("MetaARIMAModel does not return in-sample fitted values")
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        h = _check_h(h)
        base = self._select(y)
        res = self._forecast_base(base, h, _native_level(self, level))
        return self._with_intervals(res, y, X, level)

    def fit(self, y, X=None):
        """Select an ARIMA for ``y`` and store it for :meth:`predict`."""
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        self._y = y
        self._base = self._select(y)
        self._store_cs(y, X)
        return self

    def predict(self, h, X=None, level=None):
        """Forecast the series stored by :meth:`fit`."""
        if not hasattr(self, "_base"):
            raise ValueError("call fit before predict")
        h = _check_h(h)
        res = self._forecast_base(self._base, h, _native_level(self, level))
        return self._with_intervals(res, None, X, level)

    def forward(self, y, h, X=None, X_future=None, level=None, fitted: bool = False):
        """Forecast ``y``, re-selecting the ARIMA order from this window."""
        return self.forecast(y, h, X=X, X_future=X_future, level=level, fitted=fitted)

    def _select(self, y: np.ndarray):
        frame = pd.DataFrame(
            {
                "unique_id": "series",
                "ds": pd.date_range("2000-01-01", periods=len(y), freq=self.freq),
                "y": y,
            }
        )
        self.metaarima.fit(frame, freq=self.freq, seas_length=self.season_length)
        return self.metaarima.model

    def _forecast_base(self, base, h: int, level) -> dict:
        frame = base.predict(h) if level is None else base.predict(h, level=level)
        alias = base.alias
        res = {"mean": _as_1d(frame[alias], h, "forecast")}
        if not level:
            return res
        levels = sorted(level)
        for lv in reversed(levels):
            res[f"lo-{lv}"] = _as_1d(frame[f"{alias}-lo-{lv}"], h, f"lo-{lv}")
        for lv in levels:
            res[f"hi-{lv}"] = _as_1d(frame[f"{alias}-hi-{lv}"], h, f"hi-{lv}")
        return res

    def _with_intervals(self, res, y, X, level):
        if not level:
            return res
        if self.prediction_intervals is None:
            return res
        level = sorted(level)
        if y is None:
            return self._add_predict_conformal_intervals(res, level)
        return self._add_conformal_intervals(fcst=res, y=y, X=X, level=level)


def _native_level(model: MetaARIMAModel, level):
    """ARIMA intervals only when conformal intervals are not configured."""
    if not level or model.prediction_intervals is not None:
        return None
    return level


def _check_h(h) -> int:
    h = int(h)
    if h < 1:
        raise ValueError("h must be positive")
    return h


def _as_1d(values, h: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.shape != (h,):
        raise ValueError(f"expected {name} of length {h}, got {arr.shape[0]}")
    return arr
