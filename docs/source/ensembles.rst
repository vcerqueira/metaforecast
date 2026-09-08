Dynamic Ensembles
==================

Motivated by the No Free Lunch theorem,
which states that no single algorithm is optimal for all problems,
ensemble methods combine multiple models to achieve better and more
robust predictions than any individual model.

Time series forecasting particularly benefits from ensembles due to:

* Varying model performance across different periods
* Non-stationary patterns and regime changes
* Risk reduction from model selection
* State-of-the-art performance in empirical studies

This module implements dynamic ensemble strategies that adapt
weights over time to changing patterns or across different series:

**Meta-learning** — predict individual model errors with a meta-model:

- :class:`~metaforecast.ensembles.ADE`: Arbitrated Dynamic Ensemble
- :class:`~metaforecast.ensembles.MLForecastADE`: ADE with MLForecast integration

**Regret minimization** — online learning ensemble methods:

- :class:`~metaforecast.ensembles.MLewa`: Exponentially weighted averaging
- :class:`~metaforecast.ensembles.MLpol`: Polynomially weighted averaging
- :class:`~metaforecast.ensembles.MLprod`: Multiplicative production update
- :class:`~metaforecast.ensembles.BOA`: Bernstein Online Aggregation
- :class:`~metaforecast.ensembles.FixedShare`: Fixed-Share with weight redistribution
- :class:`~metaforecast.ensembles.OGD`: Online Gradient Descent
- :class:`~metaforecast.ensembles.Ridge`: Online ridge regression

**Windowing** — recent-performance-based weighting:

- :class:`~metaforecast.ensembles.Windowing`: Sliding-window performance weighting

**Static baselines** — simple combination rules:

- :class:`~metaforecast.ensembles.EqAverage`: Equal-weight averaging (with optional trimming)
- :class:`~metaforecast.ensembles.LossOnTrain`: Static weights from training loss
- :class:`~metaforecast.ensembles.BestOnTrain`: Single best model selection


Meta-Learning Ensembles
-----------------------

.. autoclass:: metaforecast.ensembles.ade.ADE
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: metaforecast.ensembles.ade.MLForecastADE
   :members:
   :undoc-members:
   :show-inheritance:


Regret Minimization Ensembles
-----------------------------

.. autoclass:: metaforecast.ensembles.mlewa.MLewa
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: metaforecast.ensembles.mlpol.MLpol
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: metaforecast.ensembles.mlprod.MLprod
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: metaforecast.ensembles.boa.BOA
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: metaforecast.ensembles.fixed_share.FixedShare
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: metaforecast.ensembles.ogd.OGD
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: metaforecast.ensembles.ridge.Ridge
   :members:
   :undoc-members:
   :show-inheritance:


Windowing
---------

.. autoclass:: metaforecast.ensembles.windowing.Windowing
   :members:
   :undoc-members:
   :show-inheritance:


Static Baselines
----------------

.. autoclass:: metaforecast.ensembles.static.EqAverage
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: metaforecast.ensembles.static.LossOnTrain
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: metaforecast.ensembles.static.BestOnTrain
   :members:
   :undoc-members:
   :show-inheritance:
