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

* Meta-learning: Using a meta-model to predict individual model errors
* Regret minimization: Exponentially or polynomially weighted averaging
* Windowing: Recent performance-based weighting

.. automodule:: metaforecast.ensembles
   :members:
   :undoc-members:
   :show-inheritance:
