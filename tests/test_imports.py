"""Smoke tests: verify all public modules import cleanly."""


def test_ensembles_import():
    from metaforecast.ensembles import (
        ADE,
        BestOnTrain,
        EqAverage,
        LossOnTrain,
        MLewa,
        MLForecastADE,
        MLpol,
        Windowing,
    )


def test_synth_import():
    from metaforecast.synth import (
        DBA,
        Jittering,
        KernelSynth,
        MagnitudeWarping,
        OnlineDataAugmentation,
        Scaling,
        SeasonalMBB,
        TimeWarping,
        TSMixup,
    )


def test_longhorizon_import():
    from metaforecast.longhorizon import MLForecastFTN


def test_version():
    import metaforecast

    assert hasattr(metaforecast, "__version__")
    assert metaforecast.__version__
