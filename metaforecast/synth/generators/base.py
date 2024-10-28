from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseTimeSeriesGenerator(ABC):
    """Abstract base class for synthetic time series generators.

    Attributes
    ----------
    REQUIRED_COLUMNS : List[str]
        Columns that must be present in input/output datasets:
        - unique_id: Series identifier
        - ds: Timestamp
        - y: Target values

    START : pd.Timestamp
        Reference start time for synthetic series

    END : pd.Timestamp
        Reference end time for synthetic series

    REQUIRES_N : bool
        Whether generator needs explicit number of series:
        - True: Pure synthetic generation
        - False: Modification of existing series

    REQUIRES_DF : bool
        Whether generator needs source dataset:
        - True: Semi-synthetic/transformation approaches
        - False: Pure synthetic generation

    """

    REQUIRED_COLUMNS = ["unique_id", "ds", "y"]
    START: pd.Timestamp
    END: pd.Timestamp
    REQUIRES_N: bool
    REQUIRES_DF: bool

    def __init__(self, alias: str):
        """Initialize semisynthetic generator with method identifier.

        Parameters
        ----------
        alias : str
            Name of the generation method being used.
            Used to track and identify different generation approaches.

        Attributes
        ----------
        alias : str
            Method identifier for the generator instance.
            Used to track and identify different generation approaches.

        counter : int
            Tracks number of synthetic series generated.
            Starts at 0 and increments with each generation.

        """
        self.alias = alias
        self.counter = 0

    @abstractmethod
    def transform(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _create_synthetic_ts(self, **kwargs):
        raise NotImplementedError

    @staticmethod
    def sample_weights_dirichlet(alpha, k):
        """Sample k weights from a Dirichlet distribution.

        Generates random weights that sum to 1 using a Dirichlet distribution.

        Parameters
        ----------
        alpha : float
            Concentration parameter of the Dirichlet distribution:

        k : int
            Number of weights to generate.
            Must be positive.

        Returns
        -------
        np.ndarray
            Array of k weights that sum to 1.0
            Shape: (k,)

        """

        # Sample from Gamma distribution
        y = np.random.gamma(alpha, 1, k)
        # Normalize the samples to get the Dirichlet weights
        weights = y / np.sum(y)

        return weights

    @classmethod
    def _assert_datatypes(cls, df: pd.DataFrame):
        """
        :param df: time series dataset with a nixtla-based structure
        """
        # Check if required columns exist
        for col in cls.REQUIRED_COLUMNS:
            assert col in df.columns, f"Column '{col}' is missing from the DataFrame"

        # Assert unique_id is of type string
        assert (
            df["unique_id"].dtype == "object"
        ), "Column 'unique_id' must be of type string"

        # Assert ds is of type pd.Timestamp
        # assert pd.api.types.is_datetime64_any_dtype(df["ds"]),
        # "Column 'ds' must be of type pd.Timestamp"

        # Assert y is numeric
        assert np.issubdtype(df["y"].dtype, np.number), "Column 'y' must be numeric"


class PureSyntheticGenerator(BaseTimeSeriesGenerator):
    """Generate synthetic time series from scratch without source data.

    Creates artificial time series using statistical models and patterns,
    without requiring existing data. Useful for:
    - Generating training data with known properties
    - Testing model behavior with controlled patterns
    - Augmenting small datasets
    - Creating benchmark datasets

    """

    START = pd.Timestamp("2000-01-01 00:00:00")
    END = pd.Timestamp("2024-01-01 00:00:00")
    REQUIRES_N = True
    REQUIRES_DF = False

    # pylint: disable=arguments-differ
    @abstractmethod
    def transform(self, n_series: int, **kwargs):
        raise NotImplementedError


class SemiSyntheticGenerator(BaseTimeSeriesGenerator):
    """Generate synthetic time series from a source dataset.

    Creates new time series that, in principle, maintain statistical properties and patterns
    from a source dataset while introducing controlled variations. This approach:
    - Preserves realistic temporal patterns
    - Maintains domain-specific characteristics
    - Allows controlled modification of specific properties
    - Generates diverse but plausible variations

    """

    REQUIRES_N = True
    REQUIRES_DF = True

    # pylint: disable=arguments-differ
    @abstractmethod
    def transform(self, df: pd.DataFrame, n_series: int, **kwargs):
        """Transform input time series into synthetic variations.

        Parameters
        ----------
        df : pd.DataFrame
            - pd.DataFrame with columns:
                - unique_id: Series identifier
                - ds: Timestamp
                - y: Target values

        n_series : int
            Number of synthetic series to generate.

        """
        raise NotImplementedError


class SemiSyntheticTransformer(BaseTimeSeriesGenerator):
    """Transform time series using specific operations while preserving structure.

    Applies controlled transformations to existing time series to create
    variations that maintain core patterns while modifying specific properties.

    """

    REQUIRES_N = False
    REQUIRES_DF = True

    def __init__(self, alias: str, rename_uids: bool = True):
        """Initialize transformer with method identifier and naming preferences.

        Parameters
        ----------
        alias : str
            Name of the transformation method to be applied.
            Used to identify and track different transformation types.
            Examples: 'scale', 'noise', 'warp'

        rename_uids : bool, default=True
            Whether to generate new identifiers for transformed series:
            - True: Create new unique_ids for transformed series
            - False: Preserve original series identifiers
            Useful when tracking relationship to source series

        Attributes
        ----------
        counter: int
            Tracks number of series transformed.
            Used for generating new unique_ids when rename_uids=True.
            Automatically increments with each transformation.

        """
        super().__init__(alias=alias)

        self.rename_uids = rename_uids

    # pylint: disable=arguments-differ
    def transform(self, df: pd.DataFrame, **kwargs):
        """Transform time series in a dataset while preserving structure.

        Applies transformation method specified by 'alias' to each series
        in the dataset. Maintains nixtla's standard format throughout
        the transformation process.

        Parameters
        ----------
        df : pd.DataFrame
            Input time series dataset with required columns:
            - unique_id: Series identifier
            - ds: Timestamp
            - y: Target values
            Must follow nixtla framework conventions

        Returns
        -------
        pd.DataFrame
            Transformed dataset with same structure:
            - Original columns preserved
            - Same temporal alignment
            - Modified y values
            - New/preserved unique_ids based on rename_uids setting

        """
        self._assert_datatypes(df)

        df_t_list = []
        for _, uid_df in df.groupby("unique_id"):
            ts_df = self._create_synthetic_ts(uid_df)
            if self.rename_uids:
                ts_df["unique_id"] = ts_df["unique_id"].apply(
                    lambda x: f"{x}_{self.alias}{self.counter}"
                )

            self.counter += 1

            df_t_list.append(ts_df)

        transformed_df = pd.concat(df_t_list).reset_index(drop=True)

        return transformed_df

    @abstractmethod
    def _create_synthetic_ts(self, df: pd.DataFrame, **kwargs):
        """_create_synthetic_ts

        Transforming a given time series

        :param df: time series with a single unique_id, plus ds, y columns
        :type df: pd.DataFrame
        """
        raise NotImplementedError
