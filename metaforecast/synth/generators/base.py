from abc import ABC, abstractmethod
from typing import Union

import torch
import numpy as np
import pandas as pd

DForTensor = Union[pd.DataFrame, torch.Tensor]


class BaseTimeSeriesGenerator(ABC):
    """
    Synthetic time series generator abstract class

    Attributes:
        REQUIRED_COLUMNS (List[str]) List of columns required in a dataset
        START (pd.Timestamp) Dummy timestamp that marks the beginning of a synthetic time series
        END (pd.Timestamp) Dummy timestamp that marks the end of a synthetic time series
        REQUIRES_N (bool) Whether the type of generator requires user input about the number of
        time series to be generated
        REQUIRES_DF (bool) Whether the type of generator requires user input about
        the source dataset
    """

    REQUIRED_COLUMNS = ['unique_id', 'ds', 'y']
    START: pd.Timestamp
    END: pd.Timestamp
    REQUIRES_N: bool
    REQUIRES_DF: bool

    def __init__(self, alias: str):
        """
        :param alias: method name
        :type alias: str

        Attributes:
        alias (str) method name
        counter (int) synthetic time series counter
        """
        self.alias = alias
        self.counter = 0

    @abstractmethod
    def transform(self, **kwargs):
        """ transform

        Create synthetic time series based on **kwargs
        """
        raise NotImplementedError

    @abstractmethod
    def _create_synthetic_ts(self, **kwargs):
        raise NotImplementedError

    @staticmethod
    def sample_weights_dirichlet(alpha, k):
        """ sample_weights_dirichlet

        Sampling weights from a Dirichlet distribution

        :param alpha: Gamma distribution parameter
        :param k: Number of samples

        :return: numpy array with weights
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
        assert df["unique_id"].dtype == "object", "Column 'unique_id' must be of type string"

        # Assert ds is of type pd.Timestamp
        # assert pd.api.types.is_datetime64_any_dtype(df["ds"]),
        # "Column 'ds' must be of type pd.Timestamp"

        # Assert y is numeric
        assert np.issubdtype(df["y"].dtype, np.number), "Column 'y' must be numeric"


class PureSyntheticGenerator(BaseTimeSeriesGenerator):
    """ PureSyntheticGenerator

    Synthetic time series generator
    Pure in the sense that realistic time series are generated without considering a source dataset.
    """

    START = pd.Timestamp('2000-01-01 00:00:00')
    END = pd.Timestamp('2024-01-01 00:00:00')
    REQUIRES_N = True
    REQUIRES_DF = False

    # pylint: disable=arguments-differ
    @abstractmethod
    def transform(self, n_series: int, **kwargs):
        raise NotImplementedError


class SemiSyntheticGenerator(BaseTimeSeriesGenerator):
    """ SemiSyntheticGenerator

    Generate synthetic time series based on a source dataset
    """
    REQUIRES_N = True
    REQUIRES_DF = True

    # pylint: disable=arguments-differ
    @abstractmethod
    def transform(self, df: DForTensor, n_series: int, **kwargs):
        """

        :param df: time series dataset, following a nixtla-based structure.
        Either a pd.DataFrame (unique_id, ds, y) or an inner tensor structure
        :type df: pd.DataFrame of torch.tensor

        :param n_series: Number of time series to be generated
        :type n_series: int
        """
        raise NotImplementedError


class SemiSyntheticTransformer(BaseTimeSeriesGenerator):
    """ SemiSyntheticTransformer

    Transform the time series in a dataset with a given operation
    """
    REQUIRES_N = False
    REQUIRES_DF = True

    def __init__(self, alias: str, rename_uids: bool = True):
        """
        :param alias: method name
        :type alias: str

        :param rename_uids: whether to rename the original unique_id's
        :type rename_uids: bool
        """
        super().__init__(alias=alias)

        self.rename_uids = rename_uids

    # pylint: disable=arguments-differ
    def transform(self, df: pd.DataFrame, **kwargs):
        """ transform

        Transform the time series of a dataset with a nixtla-based structure (unique_id, ds, y)

        :param df: time series dataset
        :type df: pd.DataFrame

        :return: Transformed dataset
        """
        self._assert_datatypes(df)

        df_t_list = []
        for _, uid_df in df.groupby('unique_id'):
            ts_df = self._create_synthetic_ts(uid_df)
            if self.rename_uids:
                ts_df['unique_id'] = \
                    ts_df['unique_id'].apply(lambda x: f'{x}_{self.alias}{self.counter}')

            self.counter += 1

            df_t_list.append(ts_df)

        transformed_df = pd.concat(df_t_list).reset_index(drop=True)

        return transformed_df

    @abstractmethod
    def _create_synthetic_ts(self, df: pd.DataFrame, **kwargs):
        """ _create_synthetic_ts

        Transforming a given time series

        :param df: time series with a single unique_id, plus ds, y columns
        :type df: pd.DataFrame
        """
        raise NotImplementedError
