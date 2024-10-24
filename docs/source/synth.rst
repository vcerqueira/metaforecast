Synthetic Time Series Generation
=================================

Data augmentation has been successfully used in various domains to increase the size of training
datasets and improve model robustness. For time series, augmentation is particularly challenging
due to the temporal dependencies and patterns that must be preserved.

This module implements several approaches for generating synthetic time series data:

* KernelSynth Creates new series using kernel-based pattern combination [1]
* DTW Barycentric Averaging (DBA): Creates new series by averaging existing ones [2]
* Jittering: Adds controlled noise variations
* Scaling: Modifies series magnitude
* MagnitudeWarping: Applies smooth amplitude changes
* TimeWarping: Creates temporal distortions
* Moving Blocks Bootstrap: Resamples temporal blocks

Each of these approaches follow one of three approaches:

* **Pure synthetic generation**: Generate synthetic time series from scratch without any source data.
* **Semi-synthetic generation**: Generate synthetic time series with reference to a source dataset.
* **Semi-synthetic transformation**: Transform time series using specific operations while preserving structure.

The module also implements a callback that applies time series augmentation techniques to each batch during model
training. This online approach creates different augmented samples in each batch


[1] Ansari, A. F., et al. (2024). "Chronos: Learning the language of time series." arXiv preprint arXiv:2403.07815.

[2] Forestier, G., et al. (2017). "Generating synthetic time series to augment sparse datasets." IEEE International Conference on Data Mining (ICDM).

[3] Um, T. T., et al. (2017). "Data augmentation of wearable sensor data for parkinson's disease monitoring."
ACM International Conference on Multimodal Interaction.

.. automodule:: metaforecast.synth
   :members:
   :undoc-members:
   :show-inheritance: