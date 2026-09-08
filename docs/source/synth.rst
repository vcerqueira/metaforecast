Synthetic Time Series Generation
=================================

Data augmentation has been successfully used in various domains to increase the size of training
datasets and improve model robustness. For time series, augmentation is particularly challenging
due to the temporal dependencies and patterns that must be preserved.

This module implements several approaches for generating synthetic time series data, organized
into three categories:

**Pure synthetic generation** — generate time series from scratch without any source data:

- :class:`~metaforecast.synth.KernelSynth`: Kernel-based pattern combination [1]
- :class:`~metaforecast.synth.SeasonalTrend`: Fourier seasonality + trend + AR residual [5]
- :class:`~metaforecast.synth.NonstationaryRegime`: Markov-switching regime changes [5]
- :class:`~metaforecast.synth.LongMemory`: ARFIMA / fractional Brownian motion [5]
- :class:`~metaforecast.synth.VolatilityEvents`: GARCH + Hawkes self-exciting spikes [5]

**Semi-synthetic generation** — generate new series with reference to a source dataset:

- :class:`~metaforecast.synth.TSMixup`: Weighted averaging of multiple series [1]
- :class:`~metaforecast.synth.DBA`: DTW Barycentric Averaging [2]

**Semi-synthetic transformation** — transform existing series while preserving structure:

- :class:`~metaforecast.synth.Jittering`: Controlled Gaussian noise injection [3]
- :class:`~metaforecast.synth.Scaling`: Amplitude scaling [3]
- :class:`~metaforecast.synth.MagnitudeWarping`: Smooth magnitude variations [3]
- :class:`~metaforecast.synth.TimeWarping`: Non-linear temporal distortions [3]
- :class:`~metaforecast.synth.SeasonalMBB`: Seasonal Moving Block Bootstrap
- :class:`~metaforecast.synth.AmplitudeModulation`: Piecewise-linear trend modulation [6]
- :class:`~metaforecast.synth.CensorAugmentation`: Quantile-based signal clipping [6]
- :class:`~metaforecast.synth.SpikeInjection`: Structured periodic spike injection [6]
- :class:`~metaforecast.synth.DominantShuffle`: Shuffle dominant frequency components [7]
- :class:`~metaforecast.synth.HomomorphicAugmentation`: Homomorphic-controlled spectral augmentation [8]

**Online augmentation** — augment data during model training:

- :class:`~metaforecast.synth.OnlineDataAugmentation`: Callback for online augmentation [4]


References
----------

[1] Ansari, A. F., et al. (2024). "Chronos: Learning the language of time series."
arXiv preprint arXiv:2403.07815.

[2] Forestier, G., et al. (2017). "Generating synthetic time series to augment sparse datasets."
IEEE International Conference on Data Mining (ICDM).

[3] Um, T. T., et al. (2017). "Data augmentation of wearable sensor data for Parkinson's
disease monitoring." ACM International Conference on Multimodal Interaction.

[4] Cerqueira, V., Santos, M., Baghoussi, Y., & Soares, C. (2024). "On-the-fly Data
Augmentation for Forecasting with Deep Learning." arXiv preprint arXiv:2404.16918.

[5] Cazaux, H., Ásgeirsson, E. I., & Stefánsson, H. (2026). "Does Synthetic Data Help?
Empirical Evidence from Deep Learning Time Series Forecasters." arXiv preprint arXiv:2605.06032.

[6] Auer, A., Bock, S., Podest, P., Klambauer, G., Klotz, D., & Hochreiter, S. (2025).
"TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context
Learning." arXiv preprint arXiv:2505.23719.

[7] Zhao, K., He, Z., Hung, A., & Zeng, D. (2024). "Dominant Shuffle: A Simple Yet
Powerful Data Augmentation for Time-series Prediction." arXiv preprint arXiv:2405.16456.

[8] Li, H., Cheng, L., Liu, X., Liu, Z., Long, L., Zhang, Y., & Dai, F. (2026).
"Homomorphic-Controlled Augmentation for Time Series Forecasting." ICASSP 2026.


Base Classes
------------

.. autoclass:: metaforecast.synth.generators.base.BaseTimeSeriesGenerator
   :members:
   :show-inheritance:
   :exclude-members: START, END, REQUIRES_N, REQUIRES_DF, GENERATOR_TYPE

.. autoclass:: metaforecast.synth.generators.base.PureSyntheticGenerator
   :members:
   :no-inherited-members:
   :show-inheritance:
   :exclude-members: START, END

.. autoclass:: metaforecast.synth.generators.base.SemiSyntheticGenerator
   :members:
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: metaforecast.synth.generators.base.SemiSyntheticTransformer
   :members:
   :no-inherited-members:
   :show-inheritance:


Pure Synthetic Generators
-------------------------

.. autoclass:: metaforecast.synth.generators.kernelsynth.KernelSynth
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: metaforecast.synth.generators.seasonal_trend.SeasonalTrend
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: metaforecast.synth.generators.nonstationary_regime.NonstationaryRegime
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: metaforecast.synth.generators.long_memory.LongMemory
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: metaforecast.synth.generators.volatility_events.VolatilityEvents
   :members:
   :undoc-members:
   :show-inheritance:


Semi-Synthetic Generators
-------------------------

.. autoclass:: metaforecast.synth.generators.tsmixup.TSMixup
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: metaforecast.synth.generators.dba.DBA
   :members:
   :undoc-members:
   :show-inheritance:


Transformers
------------

.. autoclass:: metaforecast.synth.generators.jittering.Jittering
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: metaforecast.synth.generators.scaling.Scaling
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: metaforecast.synth.generators.warping_mag.MagnitudeWarping
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: metaforecast.synth.generators.warping_time.TimeWarping
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: metaforecast.synth.generators.mbb.SeasonalMBB
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: metaforecast.synth.generators.amplitude_modulation.AmplitudeModulation
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: metaforecast.synth.generators.censor.CensorAugmentation
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: metaforecast.synth.generators.spike_injection.SpikeInjection
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: metaforecast.synth.generators.dominant_shuffle.DominantShuffle
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: metaforecast.synth.generators.homomorphic.HomomorphicAugmentation
   :members:
   :undoc-members:
   :show-inheritance:


Online Augmentation
-------------------

.. autoclass:: metaforecast.synth.callbacks.OnlineDataAugmentation
   :members:
   :undoc-members:
   :show-inheritance:
