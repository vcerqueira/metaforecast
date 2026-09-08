Configuration and Selection of Algorithms (COSEAL)
====================================================

The ``coseal`` module implements meta-learning methods for algorithm
configuration and selection, applied to time series forecasting.

MetaARIMA
---------

MetaARIMA applies meta-learning to ARIMA order selection.  Instead of
running ``AutoARIMA`` (which fits many configurations sequentially), it
uses a pre-trained meta-learner to shortlist the most promising
``(p,d,q)(P,D,Q)[m]`` configurations, then selects the best by AICc.

The two-stage workflow:

1. **Meta-training** (offline, once per frequency):

   - Collect time-series features (via ``tsfeatures``) and
     per-configuration error scores across a training corpus.
   - Train a multi-label classifier (PCA + CatBoost) that maps features
     to configuration quality probabilities.
   - (Optional) Apply MMR re-ranking for diversity.

2. **Inference** (online, per series):

   - Extract features from the new series.
   - Query the meta-learner for a shortlist of *n* promising configs.
   - Fit the shortlisted ARIMAs (via successive halving or exhaustive
     search), select by AICc.

Pre-trained models (M4 monthly, quarterly, yearly) are available in the
``experiments_extra/experiments-metaarima-main/assets/`` directory.

**References:**

[1] todo

.. autoclass:: metaforecast.coseal.metaarima.meta_arima.MetaARIMA
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
^^^^^^^^^

.. autoclass:: metaforecast.coseal.metaarima._base.MetaARIMAUtils
   :members:
   :undoc-members:

.. autofunction:: metaforecast.coseal.metaarima._base.tsfeatures_uid

.. autoclass:: metaforecast.coseal._multilabel_pca.MultiLabelPCARegressor
   :members:
   :undoc-members:
   :show-inheritance:
