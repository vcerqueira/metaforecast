"""Hygiene tests for ensemble combine/trim helpers and API consistency."""

import numpy as np
import pandas as pd

from metaforecast.ensembles import EqAverage, MLewa, Windowing


def _cv_frame(n_per_uid: int = 8) -> pd.DataFrame:
    frames = []
    for uid in ("A", "B"):
        y = np.linspace(10, 20, n_per_uid)
        frames.append(
            pd.DataFrame(
                {
                    "unique_id": uid,
                    "ds": pd.date_range("2020-01-01", periods=n_per_uid, freq="ME"),
                    "y": y,
                    "m1": y + 0.1,
                    "m2": y - 0.3,
                    "m3": y + 1.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_windowing_fit_returns_self():
    ensemble = Windowing(freq="ME")
    assert ensemble.fit(_cv_frame()) is ensemble


def test_mlewa_fit_returns_self():
    ensemble = MLewa(loss_type="square", gradient=True)
    assert ensemble.fit(_cv_frame()) is ensemble


def test_combine_forecasts_weighted_average():
    ensemble = Windowing(freq="ME")
    weights = pd.DataFrame(
        {"m1": [0.25], "m2": [0.75]},
        index=pd.Index(["A"], name="unique_id"),
    )
    fcst = pd.DataFrame(
        {
            "unique_id": ["A", "A"],
            "m1": [10.0, 20.0],
            "m2": [30.0, 40.0],
        }
    )
    out = ensemble._combine_forecasts(fcst, weights)
    assert abs(out.iloc[0] - (10 * 0.25 + 30 * 0.75)) < 1e-9
    assert abs(out.iloc[1] - (20 * 0.25 + 40 * 0.75)) < 1e-9
    assert out.name == "Windowing"


def test_apply_trim_zero_sum_falls_back_to_equal_kept():
    ensemble = Windowing(freq="ME")
    ensemble.n_models = 1
    weights = pd.DataFrame(
        {"m1": [0.0, 0.0], "m2": [1.0, 1.0]},
        index=["A", "B"],
    )
    scores = pd.DataFrame(
        {"m1": [0.1, 0.1], "m2": [0.9, 0.9]},
        index=["A", "B"],
    )
    trimmed = ensemble._apply_trim(weights, by_uid=False, scores=scores)
    assert abs(trimmed.loc["A", "m1"] - 1.0) < 1e-9
    assert abs(trimmed.loc["A", "m2"] - 0.0) < 1e-9


def test_select_best_sets_n_models_to_one():
    ensemble = Windowing(freq="ME", select_best=True)
    ensemble.fit(_cv_frame())
    assert ensemble.n_models == 1
    assert ensemble.alias == "BLAST"


def test_eqaverage_select_by_uid_alias():
    ensemble = EqAverage(select_by_uid=False)
    assert ensemble.weight_by_uid is False


def test_weight_by_uid_defaults_true():
    assert Windowing(freq="ME").weight_by_uid is True
    assert MLewa(loss_type="square", gradient=True).weight_by_uid is True


def test_window_size_aliases():
    expected = {
        "h": 48,
        "H": 48,
        "B": 10,
        "W-SUN": 16,
        "W-MON": 16,
        "BME": 12,
        "BMS": 12,
        "QE-DEC": 4,
        "BQE": 4,
        "BQS": 4,
        "YE": 6,
        "YS": 6,
        "A": 6,
        "AS": 6,
    }
    for freq, window_size in expected.items():
        assert Windowing(freq=freq).window_size == window_size


def test_unknown_freq_raises_value_error():
    try:
        Windowing(freq="foo")
    except ValueError as exc:
        assert "Unknown freq" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_assert_fcst_requires_unique_id():
    ensemble = Windowing(freq="ME")
    try:
        ensemble.predict(pd.DataFrame({"ds": [1], "m1": [0.0]}))
    except ValueError as exc:
        assert "unique_id" in str(exc)
    else:
        raise AssertionError("expected ValueError")
