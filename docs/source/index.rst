metaforecast
=============

**metaforecast** is a Python package for time series forecasting using
meta-learning and data-centric techniques.

It implements various techniques to enhance forecasting performance through
model combination, data augmentation, and adaptive learning, building upon
`Nixtla's <https://github.com/Nixtla>`_ ecosystem of state-of-the-art
forecasting methods.


Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install metaforecast


Modules
~~~~~~~

Dynamic Ensembles
-----------------

Combines multiple forecasting models using adaptive weighting strategies:

- Online learning with exponential and polynomial weights
- Performance-based dynamic model selection and trimming
- Predicted weights based on meta-learning

:doc:`Go to API reference → <ensembles>`


Synthetic Time Series Generation
---------------------------------

Creates synthetic time series data for augmentation and testing:

- *Pure* synthetic generation through kernel methods
- Semi-synthetic generation preserving the patterns of a source dataset
- Transformation-based augmentation (jittering, scaling, warping, bootstrap)
- Online augmentation during model training

:doc:`Go to API reference → <synth>`


Long-Horizon Meta-Learning
---------------------------

Improves multi-step forecasting accuracy through instance-based approaches:

- Trajectory-based nearest neighbor matching (FTN)

:doc:`Go to API reference → <longhorizon>`


.. toctree::
   :maxdepth: 2
   :caption: API Reference

   ensembles
   synth
   longhorizon

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   notebooks
