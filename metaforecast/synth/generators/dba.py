import warnings

import numpy as np
import pandas as pd
from tslearn.barycenters import dtw_barycenter_averaging_subgradient as dtw
from tslearn.barycenters.dba import ConvergenceWarning

from metaforecast.synth.generators.base import SemiSyntheticGenerator

warnings.filterwarnings('ignore', category=ConvergenceWarning)


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
    >>> from metaforecast.utils.data import DataUtils
    >>>
    >>> # Loading and preparing data
    >>> df, *_ = M3.load('.', group='Monthly')
    >>>
    >>> horizon = 12
    >>> train, test = DataUtils.train_test_split(df, horizon)
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

    def __init__(self, max_n_uids: int, dirichlet_alpha: float = 1.0, max_iter: int = 10, tol: float = 1e-3):
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

    # pylint: disable=unused-variable
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

        unq_uids = df["unique_id"].unique()

        if n_series < 0:
            n_series = len(unq_uids)

        dataset = []
        for _ in range(n_series):
            n_uids = np.random.randint(1, self.max_n_uids + 1)

            selected_uids = np.random.choice(unq_uids, n_uids, replace=True).tolist()

            df_uids = df.query("unique_id == @selected_uids")

            ts_df = self._create_synthetic_ts(df_uids)
            ts_df["unique_id"] = f"{self.alias}_{self.counter}"
            self.counter += 1

            dataset.append(ts_df)

        synth_df = pd.concat(dataset).reset_index(drop=True)

        return synth_df

    # pylint: disable=arguments-differ
    def _create_synthetic_ts(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Apply DBA to a time series dataset

        :param df: time series dataset with a sample of unique_id's
        :return: pd.DataFrame with synthetic time series
        """
        y_list = [y["y"].values for _, y in df.groupby("unique_id")]
        uid_size = df["unique_id"].value_counts()

        ds = df.query(f'unique_id=="{uid_size.index[0]}"')["ds"].values

        w = self.sample_weights_dirichlet(1, len(y_list))

        synth_y = dtw(X=y_list, weights=w, max_iter=self.max_iter, tol=self.tol)
        synth_y = synth_y.flatten()

        synth_df = pd.DataFrame({"ds": ds[: len(synth_y)], "y": synth_y})

        return synth_df
