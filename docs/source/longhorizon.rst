Long-Horizon Forecasting
==========================

Multi-step prediction is a key challenge in time series forecasting. However, forecasting
accuracy typically decreases as predictions are made further into the future. This is
caused by both decreasing predictability and error propagation along the horizon.

This module implements methods specifically designed to improve long-horizon forecasting
accuracy:

- :class:`~metaforecast.longhorizon.MLForecastFTN`: Forecast Trajectory Neighbors (FTN),
  a meta-learning strategy that can be integrated with any forecasting model. FTN works
  by using training observations to correct errors in multi-step predictions through
  nearest neighbor matching of forecast trajectories [1].


References
----------

[1] Cerqueira, V., Torgo, L., & Bontempi, G. (2024). "Instance-based meta-learning for
conditionally dependent univariate multistep forecasting."
International Journal of Forecasting.


API Reference
-------------

.. autoclass:: metaforecast.longhorizon.ftn.ForecastTrajectoryNeighbors
   :members:
   :show-inheritance:

.. autoclass:: metaforecast.longhorizon.ftn.MLForecastFTN
   :members:
   :undoc-members:
   :show-inheritance:
