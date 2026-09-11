import warnings

import numpy as np
import pandas as pd
from tslearn.barycenters import dtw_barycenter_averaging_subgradient as dtw

from metaforecast.synth.generators.base import SemiSyntheticGenerator

try:
    from tslearn.barycenters.dba import ConvergenceWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except (ImportError, TypeError, AssertionError):
    pass


class DBA(SemiSyntheticGenerator):
    """Generate synthetic time series using DTW Barycentric Averaging.

    Creates new time series by computing weighted averages of existing
    series using Dynamic Time Warping (DTW) alignment. This method:
    - Preserves temporal patterns while allowing variations
    - Creates realistic interpolations between series
    - Maintains temporal dependencies

    Based on the method described in [1].

    References
    ----------
    [1] Forestier, G., Petitjean, F., Dau, H.A., Webb, G.I.,
    Keogh, E. (2017). "Generating synthetic time series to
    augment sparse datasets." In IEEE International Conference
    on Data Mining (ICDM), pp. 865-870.

    Examples
    --------
    >>> import pandas as pd
    >>> from datasetsforecast.m3 import M3
    >>> from neuralforecast import NeuralForecast
    >>> from neuralforecast.models import NHITS
    >>>
    >>> from metaforecast.synth import DBA
    >>> from metaforecast.evaluation.cv import SeriesWiseSplit
    >>>
    >>> # Loading and preparing data
    >>> df, *_ = M3.load('.', group='Monthly')
    >>>
    >>> horizon = 12
    >>> train, test = SeriesWiseSplit.time_wise_split(df, horizon)
    >>>
    >>> # Data augmentation
    >>> tsgen = DBA(max_n_uids=10)
    >>> ## Create 100 time series
    >>> synth_df = tsgen.transform(train, 100)
    >>> ## Concat the synthetic dataset with the original training data
    >>> train_aug = pd.concat([train, synth_df])
    >>>
    >>> # Setting up NHITS
    >>> models = [NHITS(input_size=horizon, h=horizon, accelerator='cpu')]
    >>> nf = NeuralForecast(models=models, freq='M')
    >>>
    >>> # Fitting NHITS on the augmented data
    >>> nf.fit(df=train_aug)
    >>>
    >>> # Forecasting on the original dataset
    >>> fcst = nf.predict(df=train)
    """

    def __init__(
        self, max_n_uids: int, dirichlet_alpha: float = 1.0, max_iter: int = 10, tol: float = 1e-3
    ):
        """Initialize DBA generator with sampling parameters.

        Parameters
        ----------
        max_n_uids : int
            Maximum number of source series to combine in each generation.
            Must be positive.

        dirichlet_alpha : float, default=1.0
            Concentration parameter for Dirichlet distribution used in
            generating combination weights:

        """
        super().__init__(alias="DBA")

        self.max_n_uids = max_n_uids
        self.dirichlet_alpha = dirichlet_alpha
        self.max_iter = max_iter
        self.tol = tol

    def transform(self, df: pd.DataFrame, n_series: int = -1, **kwargs):
        """Generate synthetic time series using DTW Barycentric Averaging.

        Creates new time series by computing DTW-based weighted averages
        from randomly selected subsets of source series. Maintains the
        nixtla framework structure throughout generation.

        Parameters
        ----------
        df : pd.DataFrame
            Source time series dataset with required columns:
            - unique_id: Series identifier
            - ds: Timestamp
            - y: Target values
            Must follow nixtla framework conventions

        n_series : int, default=-1
            Number of synthetic series to generate:
            - If -1: Generate same number as source dataset
            - If positive: Generate specified number

        Returns
        -------
        pd.DataFrame
            Generated synthetic series with same structure:
            - New unique_ids: f"DBA_{i}" for i in range(n_series)
            - Same temporal alignment as source
            - Averaged y values from DBA combinations

        """

        self._assert_datatypes(df)

        y_by_uid, ds_by_uid, lengths = self._index_series(df)
        n_pool = len(y_by_uid)
        if n_pool == 0:
            raise ValueError("df contains no series to average")

        if n_series < 0:
            n_series = n_pool

        uid_chunks = []
        ds_chunks = []
        y_chunks = []
        for _ in range(n_series):
            n_uids = np.random.randint(1, self.max_n_uids + 1)
            chosen = np.unique(np.random.choice(n_pool, n_uids, replace=True))
            ds, y = self._average_series(y_by_uid, ds_by_uid, lengths, chosen)
            uid_chunks.append(np.full(len(y), f"{self.alias}_{self.counter}", dtype=object))
            ds_chunks.append(ds)
            y_chunks.append(y)
            self.counter += 1

        return pd.DataFrame(
            {
                self.id_col: np.concatenate(uid_chunks),
                self.time_col: np.concatenate(ds_chunks),
                self.target_col: np.concatenate(y_chunks),
            }
        )

    def _index_series(self, df: pd.DataFrame):
        """Split the panel into per-series NumPy arrays (one groupby)."""
        y_by_uid = []
        ds_by_uid = []
        lengths = []
        for _, uid_df in df.groupby(self.id_col, sort=False):
            y_by_uid.append(uid_df[self.target_col].to_numpy())
            ds_by_uid.append(uid_df[self.time_col].to_numpy())
            lengths.append(len(uid_df))
        return y_by_uid, ds_by_uid, np.asarray(lengths)

    def _create_synthetic_ts(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        y_by_uid, ds_by_uid, lengths = self._index_series(df)
        ds, y = self._average_series(y_by_uid, ds_by_uid, lengths, np.arange(len(y_by_uid)))
        return pd.DataFrame({self.time_col: ds, self.target_col: y})

    def _average_series(self, y_by_uid, ds_by_uid, lengths, chosen: np.ndarray):
        """DTW barycenter of the series indexed by ``chosen``."""
        y_list = [y_by_uid[i] for i in chosen]
        longest = chosen[int(np.argmax(lengths[chosen]))]
        weights = self.sample_weights_dirichlet(self.dirichlet_alpha, len(y_list))
        synth_y = dtw(X=y_list, weights=weights, max_iter=self.max_iter, tol=self.tol).flatten()
        return ds_by_uid[longest][: len(synth_y)], synth_y
