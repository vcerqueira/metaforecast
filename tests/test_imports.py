"""Smoke tests: verify all public modules import cleanly."""

import importlib


def test_ensembles_import():
    mod = importlib.import_module("metaforecast.ensembles")
    assert hasattr(mod, "ADE")
    assert hasattr(mod, "MLewa")
    assert hasattr(mod, "Windowing")


def test_synth_import():
    mod = importlib.import_module("metaforecast.synth")
    assert hasattr(mod, "KernelSynth")
    assert hasattr(mod, "Jittering")
    assert hasattr(mod, "OnlineDataAugmentation")


def test_longhorizon_import():
    mod = importlib.import_module("metaforecast.longhorizon")
    assert hasattr(mod, "MLForecastFTN")


def test_version():
    import metaforecast

    assert hasattr(metaforecast, "__version__")
    assert metaforecast.__version__
