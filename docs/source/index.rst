metaforecast
=============

metaforecast is a Python package for time series forecasting using meta-learning and data-centric techniques.

This package implements various techniques to enhance
forecasting performance through model combination, data augmentation, and adaptive
learning, building upon Nixtla's awesome ecosystem of state-of-the-art forecasting methods.

Dynamic Ensembles
~~~~~~~~~~~~~~~~~~~~~
Combines multiple forecasting models using adaptive weighting strategies:

- Online learning with exponential and polynomial weights
- Performance-based dynamic model selection and trimming
- Predicted weights based on meta-learning

Synthetic Time Series Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Creates synthetic time series data for augmentation and testing:

* *Pure* synthetic generation through kernel methods
* Semi-synthetic generation preserving the patterns of a source dataset
* Transformation-based augmentation, by applying relevant operations to a given dataset
* Online augmentation during model training

Long-Horizon Meta-Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Improves multi-step forecasting accuracy through instance-based approaches:

- Trajectory-based nearest neighbor matching


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   ensembles
   synth
   longhorizon
   notebooks
