"""Evaluation utilities for metaforecast.

Submodules
----------
cv : Series-wise cross-validation splitters.
aspects : Aspect-based forecast evaluation (ModelRadar).
"""

from metaforecast.evaluation.aspects import ModelRadar, RopeAnalysis
from metaforecast.evaluation.cv import SeriesWiseSplit

__all__ = ["ModelRadar", "RopeAnalysis", "SeriesWiseSplit"]
