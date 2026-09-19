"""Tests for ModelRadar evaluation API, scoring cache, and aspect helpers."""

import numpy as np
import pandas as pd
import pytest
from utilsforecast.losses import mae, smape

from metaforecast.evaluation.aspects import ModelRadar, RopeAnalysis
from metaforecast.evaluation.aspects.radar import BaseModelRadar


def _cv_frame(
    n_uid: int = 5,
    h: int = 4,
    n_cutoffs: int = 2,
    *,
    extra_model: bool = True,
) -> pd.DataFrame:
    rows = []
    uids = [f"S{i}" for i in range(n_uid)]
    for uid_i, uid in enumerate(uids):
        for cutoff_i in range(n_cutoffs):
            cutoff = pd.Timestamp("2020-01-31") + pd.offsets.MonthEnd(cutoff_i)
            for step in range(1, h + 1):
                ds = cutoff + pd.offsets.MonthEnd(step)
                y = 10.0 + uid_i + step
                row = {
                    "unique_id": uid,
                    "ds": ds,
                    "cutoff": cutoff,
                    "horizon": step,
                    "y": y,
                    "m1": y + 0.1 * (uid_i + 1),
                    "m2": y + 0.5 * (uid_i + 1),
                    "SeasonalNaive": y + 2.0 + uid_i,
                    "NHITS": y + 0.3,
                }
                if extra_model:
                    row["yearly"] = y + 0.2
                rows.append(row)
    return pd.DataFrame(rows)


def _radar(cv: pd.DataFrame | None = None, **kwargs) -> ModelRadar:
    cv = _cv_frame() if cv is None else cv
    defaults: dict = dict(metrics=[smape], rope=10)
    if "model_names" not in kwargs:
        defaults.update(
            hardness_reference="SeasonalNaive",
            rope_reference="NHITS",
        )
    defaults.update(kwargs)
    return ModelRadar(cv_df=cv, **defaults)


def test_model_order_is_lazy():
    radar = _radar()
    assert radar._uid_scores is None
    assert radar._model_order is None
    order = radar.model_order
    assert radar._uid_scores is not None
    assert order == radar.evaluate().sort_values().index.tolist()
    assert set(order) <= set(radar.models)


def test_uid_scores_cached_and_evaluate_by_uid():
    radar = _radar()
    err = radar.evaluate_by_uid()
    assert radar._uid_scores is err
    assert list(err.index) == [f"S{i}" for i in range(5)]
    overall = radar.evaluate()
    assert overall.name == "Overall"
    pd.testing.assert_series_equal(overall, err.mean().rename("Overall"), check_names=True)
    pd.testing.assert_frame_equal(err, radar.uid_scores())


def test_custom_column_names_forwarded():
    cv = _cv_frame().rename(columns={"unique_id": "item_id", "ds": "time", "y": "target"})
    radar = ModelRadar(
        cv_df=cv,
        metrics=[smape],
        model_names=["m1", "m2"],
        id_col="item_id",
        time_col="time",
        target_col="target",
    )
    overall = radar.evaluate()
    err = radar.evaluate_by_uid()
    assert list(overall.index) == ["m1", "m2"]
    assert err.index.name == "item_id"
    assert not err.isna().any().any()


def test_autodetect_keeps_yearly_model():
    radar = ModelRadar(cv_df=_cv_frame(), metrics=[smape])
    assert "yearly" in radar.models
    assert "y" not in radar.models
    assert "horizon" not in radar.models


def test_existing_horizon_is_preserved():
    cv = _cv_frame()
    cv["horizon"] = 99
    radar = _radar(cv, model_names=["m1", "m2"])
    assert (radar.cv_df["horizon"] == 99).all()


def test_integer_time_col_not_converted_to_datetime():
    cv = _cv_frame().drop(columns=["horizon"])
    cv["ds"] = np.arange(len(cv))
    radar = _radar(cv, model_names=["m1", "m2"])
    assert pd.api.types.is_integer_dtype(radar.cv_df["ds"])
    assert "horizon" in radar.cv_df.columns


def test_hard_uids_and_facade_methods():
    radar = _radar(hardness_quantile=0.6)
    err = radar.evaluate_by_uid()
    hard_list = radar.hard_uids()
    hard_df = radar.hard_uids(return_df=True)
    assert isinstance(hard_list, list)
    assert len(hard_list) >= 1
    assert list(hard_df.index) == hard_list
    on_hard = radar.evaluate_on_hard()
    assert on_hard.name == "On Hard"
    pd.testing.assert_series_equal(on_hard, err.loc[hard_list].mean().rename("On Hard"))
    cvar = radar.expected_shortfall()
    assert cvar.name == "Exp. Shortfall"
    assert radar.hardness is radar.uid_accuracy


def test_winning_ratios_and_ratios_reference_alias():
    via_alias = _radar(rope_reference=None, ratios_reference="NHITS")
    via_pref = _radar()
    pd.testing.assert_frame_equal(via_alias.winning_ratios(), via_pref.winning_ratios())
    rope_hard = via_pref.winning_ratios(on_hard=True)
    assert not rope_hard.empty
    assert "draw" in rope_hard.columns


def test_conflicting_rope_references_raise():
    with pytest.raises(ValueError, match="both set"):
        _radar(rope_reference="NHITS", ratios_reference="m1")


def test_invalid_reference_raises():
    with pytest.raises(ValueError, match="hardness_reference"):
        _radar(hardness_reference="missing")


def test_keep_metrics_does_not_average():
    radar = _radar(metrics=[smape, mae], model_names=["m1", "m2"])
    by_metric = radar.evaluate(keep_metrics=True)
    assert list(by_metric.index) == ["smape", "mae"] or set(by_metric.index) == {"smape", "mae"}
    by_uid_metric = radar.evaluate_by_uid(keep_metrics=True)
    assert by_uid_metric.index.nlevels == 2


def test_rope_inclusive_boundary_and_zero_reference():
    rope = RopeAnalysis(rope=10, reference="ref")
    scores = pd.DataFrame(
        {"ref": [10.0, 10.0, 10.0], "other": [11.0, 9.0, 10.0]},
        index=["a", "b", "c"],
    )
    probs = rope.get_winning_ratios(scores)
    assert probs.loc["other", "draw"] == pytest.approx(1.0)

    zero_ref = pd.DataFrame({"ref": [0.0, 1.0], "other": [1.0, 1.0]})
    probs_zero = rope.get_winning_ratios(zero_ref)
    assert probs_zero.loc["other"].sum() == pytest.approx(1.0)

    with pytest.raises(ValueError, match="not set"):
        RopeAnalysis(rope=10, reference=None).get_winning_ratios(scores)


def test_horizon_cumulative_vs_per_step():
    radar = _radar(model_names=["m1", "m2"])
    cumulative = radar.evaluate_by_horizon(cumulative=True)
    per_step = radar.evaluate_by_horizon(cumulative=False)
    assert "horizon" in cumulative.columns
    assert len(cumulative) == len(per_step) == 4
    assert not cumulative.drop(columns="horizon").equals(per_step.drop(columns="horizon"))


def test_horizon_bounds_use_min_max_step():
    radar = _radar(model_names=["m1", "m2"])
    bounds = radar.evaluate_by_horizon_bounds()
    first = radar.evaluate(radar.cv_df.loc[radar.cv_df["horizon"] == 1])
    last = radar.evaluate(radar.cv_df.loc[radar.cv_df["horizon"] == 4])
    pd.testing.assert_series_equal(bounds["First horizon"], first, check_names=False)
    pd.testing.assert_series_equal(bounds["Last horizon"], last, check_names=False)


def test_anomaly_vectorized_matches_subset():
    cv = _cv_frame()
    cv["is_anomaly"] = 0
    cv.loc[cv["unique_id"].eq("S0") & cv["horizon"].eq(1), "is_anomaly"] = 1
    radar = _radar(cv, model_names=["m1", "m2"])
    obs = radar.evaluate_by_anomaly(mode="observations")
    series = radar.evaluate_by_anomaly(mode="series")
    expected_obs = radar.evaluate(radar.cv_df.loc[radar.cv_df["is_anomaly"] > 0], keep_uids=True)
    expected_series = radar.evaluate(
        radar.cv_df.loc[radar.cv_df["unique_id"].eq("S0")], keep_uids=True
    )
    pd.testing.assert_frame_equal(obs, expected_obs)
    pd.testing.assert_frame_equal(series, expected_series)
    empty = radar.evaluate_by_anomaly(cv=radar.cv_df.assign(is_anomaly=0), mode="observations")
    assert empty.empty
    assert list(empty.columns) == ["m1", "m2"]


def test_aspect_table_index_and_columns():
    cv = _cv_frame()
    cv["stationarity"] = np.where(cv["unique_id"].isin(["S0", "S1"]), "stationary", "unit_root")
    radar = _radar(
        cv,
        model_names=["m1", "m2", "NHITS", "SeasonalNaive"],
        hardness_reference="SeasonalNaive",
        rope_reference="NHITS",
    )
    table = radar.aspect_table(group_cols=["stationarity"])
    assert list(table.index) == radar.models or set(table.index) == set(radar.models)
    for col in ("Overall", "Exp. Shortfall", "First horizon", "Last horizon", "On Hard"):
        assert col in table.columns
    assert "stationary" in table.columns
    assert "unit_root" in table.columns


def test_base_model_radar_not_exported():
    import metaforecast.evaluation.aspects as aspects

    assert not hasattr(aspects, "BaseModelRadar")
    assert issubclass(BaseModelRadar, object)


def test_freq_col_missing_raises():
    radar = _radar(model_names=["m1"])
    with pytest.raises(KeyError, match="Frequency"):
        radar.evaluate_by_horizon(group_by_freq=True)
