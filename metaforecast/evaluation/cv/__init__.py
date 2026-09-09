"""Series-wise cross-validation for time series forecasting.

Splitters operate on the unique-ID dimension: in each fold a subset of
series is used for training and a disjoint subset for testing.  The
temporal split (history vs. forecast horizon) is handled by the
forecasting framework (NeuralForecast / StatsForecast).
"""

from metaforecast.evaluation.cv._base import SeriesWiseSplit
from metaforecast.evaluation.cv.bootstrap import (
    SeriesWiseBootstrap,
    SeriesWiseRepeatedBootstrap,
)
from metaforecast.evaluation.cv.holdout import (
    SeriesWiseHoldout,
    SeriesWiseMonteCarlo,
    SeriesWiseRepeatedHoldout,
)
from metaforecast.evaluation.cv.kfold import (
    SeriesWiseKFold,
    SeriesWiseRepeatedKFold,
)
from metaforecast.evaluation.cv.neuralforecast_ext import SeriesWiseNeuralForecast

__all__ = [
    "SeriesWiseBootstrap",
    "SeriesWiseHoldout",
    "SeriesWiseKFold",
    "SeriesWiseMonteCarlo",
    "SeriesWiseNeuralForecast",
    "SeriesWiseRepeatedBootstrap",
    "SeriesWiseRepeatedHoldout",
    "SeriesWiseRepeatedKFold",
    "SeriesWiseSplit",
]
