"""StatsForecast adapter for a meta-fitted MetaARIMA."""

import numpy as np
import pandas as pd
import pytest
from statsforecast import StatsForecast

from metaforecast.coseal import MetaARIMA, MetaARIMAModel


def test_rejects_unfitted_and_wrong_type():
    with pytest.raises(TypeError, match="MetaARIMA"):
        MetaARIMAModel(object())

    meta = MetaARIMA(model=object(), freq="YE", season_length=1, n_trials=1)
    with pytest.raises(ValueError, match="meta-fitted"):
        MetaARIMAModel(meta)


def test_forecast_rejects_fitted_values_and_bad_horizon():
    meta = MetaARIMA(model=object(), freq="YE", season_length=1, n_trials=1)
    meta.is_fit = True
    model = MetaARIMAModel(meta)

    with pytest.raises(ValueError, match="fitted values"):
        model.forecast(np.ones(8), h=2, fitted=True)
    with pytest.raises(ValueError, match="positive"):
        model.forecast(np.ones(8), h=0)
    with pytest.raises(ValueError, match="fit before predict"):
        model.predict(h=2)


@pytest.fixture(scope="module")
def yearly_meta():
    return MetaARIMA.load(freq="YE")


def _panel(freq: str, n: int = 36, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for uid, slope in (("a", 0.3), ("b", -0.15)):
        y = 10 + slope * np.arange(n) + rng.normal(scale=0.4, size=n)
        frames.append(
            pd.DataFrame(
                {
                    "unique_id": uid,
                    "ds": pd.date_range("2000-01-01", periods=n, freq=freq),
                    "y": y,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_fit_predict_keeps_a_base_model_per_series(yearly_meta):
    sf = StatsForecast(
        models=[MetaARIMAModel(yearly_meta)],
        freq=yearly_meta.freq,
        n_jobs=1,
    )
    sf.fit(_panel(yearly_meta.freq))

    first, second = sf.fitted_[0, 0], sf.fitted_[1, 0]
    assert first._base is not second._base
    assert first.metaarima is second.metaarima

    fcst = sf.predict(h=4, level=[80])
    assert len(fcst) == 8
    assert fcst["unique_id"].nunique() == 2
    for col in ("MetaARIMA", "MetaARIMA-lo-80", "MetaARIMA-hi-80"):
        assert col in fcst.columns, list(fcst.columns)
        assert np.isfinite(fcst[col]).all()
    assert (fcst["MetaARIMA-lo-80"] <= fcst["MetaARIMA"]).all()
    assert (fcst["MetaARIMA"] <= fcst["MetaARIMA-hi-80"]).all()


def test_forecast_one_series(yearly_meta):
    train = _panel(yearly_meta.freq).query("unique_id == 'a'")
    sf = StatsForecast(
        models=[MetaARIMAModel(yearly_meta)],
        freq=yearly_meta.freq,
        n_jobs=1,
    )
    fcst = sf.forecast(df=train, h=3)
    assert len(fcst) == 3
    assert "MetaARIMA" in fcst.columns
    assert np.isfinite(fcst["MetaARIMA"]).all()
