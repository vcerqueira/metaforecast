Evaluation
==========

Series-Wise Cross-Validation
-----------------------------

Cross-validation splitters that operate on the **series** (unique ID)
dimension.  In each fold a subset of series is used for training and a
disjoint subset for testing; the temporal split (history vs. forecast
horizon) is handled separately by the forecasting framework.


Base Class
^^^^^^^^^^

.. autoclass:: metaforecast.evaluation.cv.SeriesWiseSplit
   :members:
   :undoc-members:
   :show-inheritance:


Holdout Splitters
^^^^^^^^^^^^^^^^^

.. autoclass:: metaforecast.evaluation.cv.SeriesWiseHoldout
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: metaforecast.evaluation.cv.SeriesWiseRepeatedHoldout
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: metaforecast.evaluation.cv.SeriesWiseMonteCarlo
   :members:
   :undoc-members:
   :show-inheritance:


K-Fold Splitters
^^^^^^^^^^^^^^^^

.. autoclass:: metaforecast.evaluation.cv.SeriesWiseKFold
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: metaforecast.evaluation.cv.SeriesWiseRepeatedKFold
   :members:
   :undoc-members:
   :show-inheritance:


Bootstrap Splitters
^^^^^^^^^^^^^^^^^^^

.. autoclass:: metaforecast.evaluation.cv.SeriesWiseBootstrap
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: metaforecast.evaluation.cv.SeriesWiseRepeatedBootstrap
   :members:
   :undoc-members:
   :show-inheritance:


NeuralForecast Extension
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: metaforecast.evaluation.cv._custom_nf.SeriesWiseNeuralForecast
   :members:
   :undoc-members:
   :show-inheritance:


Aspect-Based Evaluation (ModelRadar)
------------------------------------

Slice forecast error across overall mean, unique ID, horizon, anomalies,
and arbitrary groups.  Also reports ROPE win/draw/loss vs. a reference
model and tail risk (expected shortfall) on hard series.

**References:**

Cerqueira, V., Roque, L., & Soares, C. (2025). Modelradar: aspect-based
forecast evaluation. *Machine Learning*, 114(10), 229.

.. autoclass:: metaforecast.evaluation.aspects.ModelRadar
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: metaforecast.evaluation.aspects.ModelRadarAcrossId
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: metaforecast.evaluation.aspects.RopeAnalysis
   :members:
   :undoc-members:
   :show-inheritance:
