"""Constructor-level tests for ADE (no meta-model training)."""

import numpy as np

from metaforecast.ensembles import ADE


def test_meta_lags_integer_becomes_range():
    ensemble = ADE(freq="ME", h=12, meta_lags=6)
    assert ensemble.meta_lags == list(range(1, 7))


def test_meta_lags_numpy_integer_becomes_range():
    ensemble = ADE(freq="ME", h=12, meta_lags=np.int64(4))
    assert ensemble.meta_lags == list(range(1, 5))


def test_meta_lags_none_uses_frequency_window():
    ensemble = ADE(freq="ME", h=12)
    assert ensemble.meta_lags == list(range(1, 13))


def test_meta_lags_list_is_kept():
    lags = [1, 3, 8]
    ensemble = ADE(freq="ME", h=12, meta_lags=lags)
    assert ensemble.meta_lags == lags


def test_default_meta_models_are_not_shared():
    a = ADE(freq="ME", h=12)
    b = ADE(freq="ME", h=12)
    assert a.meta_model is not b.meta_model
