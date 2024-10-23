Long-horizon forecasting
==========================

Multi-step prediction is a key challenge in time series forecasting. However, forecasting
accuracy typically decreases as predictions are made further into the future. This is
caused by both decreasing predictability and error propagation along the horizon.

This module implements methods specifically designed to improve long-horizon forecasting
accuracy:

* Forecast Trajectory Neighbors (FTN): A meta-learning strategy that can be integrated with any forecasting model. FTN works by using training observations to correct errors in multi-step predictions through nearest neighbor matching of forecast trajectories [1].


[1] Cerqueira, V., Torgo, L., & Bontempi, G. (2024). "Instance-based meta-learning for conditionally dependent
univariate multistep forecasting." International Journal of Forecasting.

.. automodule:: metaforecast.longhorizon
   :members:
   :undoc-members:
   :show-inheritance:
