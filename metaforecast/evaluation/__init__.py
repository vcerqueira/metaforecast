"""Evaluation utilities for metaforecast.

Submodules
----------
cv : Series-wise cross-validation splitters.
"""

from metaforecast.evaluation.cv import SeriesWiseSplit

__all__ = ["SeriesWiseSplit"]
