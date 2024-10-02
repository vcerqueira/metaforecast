from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Union

import torch
import numpy as np
import pandas as pd

# from metaforecast.utils.data import DataUtils

DForTensor = Union[pd.DataFrame, torch.Tensor]


class BaseTimeSeriesGenerator(ABC):
    REQUIRES_N: bool
    REQUIRES_DF: bool

    def __init__(self, alias: str):
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
        # Sample from Gamma distribution
        y = np.random.gamma(alpha, 1, k)
        # Normalize the samples to get the Dirichlet weights
        weights = y / np.sum(y)

        return weights


class PureSyntheticGenerator(BaseTimeSeriesGenerator):
    START = pd.Timestamp('2000-01-01 00:00:00')
    END = pd.Timestamp('2024-01-01 00:00:00')

    REQUIRES_N = True
    REQUIRES_DF = False

    @abstractmethod
    def transform(self, n_series: int):
        raise NotImplementedError


class SemiSyntheticGenerator(BaseTimeSeriesGenerator):
    REQUIRES_N = True
    REQUIRES_DF = True

    @abstractmethod
    def transform(self, df: DForTensor, n_series: int):
        raise NotImplementedError


class SemiSyntheticTransformer(BaseTimeSeriesGenerator):
    REQUIRES_N = False
    REQUIRES_DF = True

    def __init__(self, alias: str, rename_uids: bool = True):
        super().__init__(alias=alias)

        self.rename_uids = rename_uids

    def transform(self, df: pd.DataFrame):
        df_t_list = []
        for uid, uid_df in df.groupby('unique_id'):
            ts_df = self._create_synthetic_ts(uid_df)
            if self.rename_uids:
                ts_df['unique_id'] = ts_df['unique_id'].apply(lambda x: f'{x}_{self.alias}{self.counter}')

            self.counter += 1

            df_t_list.append(ts_df)

        transformed_df = pd.concat(df_t_list).reset_index(drop=True)

        return transformed_df

    @abstractmethod
    def _create_synthetic_ts(self, df: pd.DataFrame):
        raise NotImplementedError


# class TSDataGenerator(BaseTSGenerator):
#
#     def __init__(self, test_size: int = 0):
#         super().__init__()
#
#         self.test_size = test_size
#
#     def transform(self, df: pd.DataFrame, **kwargs):
#         """
#         :param df: (unique_id, ds, value)
#         """
#
#         if self.test_size > 0:
#             train, test = DataUtils.train_test_split(df, self.test_size)
#
#             synth_train = self._transform(train)
#
#             synth_test = test.copy()
#             synth_test['unique_id'] = test['unique_id'].apply(lambda x: f'Synth_{x}')
#
#             synth_df = pd.concat([synth_train, synth_test]).reset_index(drop=True)
#         else:
#             synth_df = self._transform(df)
#
#         augmented_df = pd.concat([synth_df, df]).reset_index(drop=True)
#         augmented_df = augmented_df.sort_values(['unique_id', 'ds']).reset_index(drop=True)
#
#         return augmented_df
#
#     def _transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
#         """
#         :param df: (unique_id, ds, value)
#         """
#
#         grouped_df = df.groupby('unique_id')
#
#         synth_l = []
#         for g, uid_df in grouped_df:
#             synth_df = self._create_synthetic_ts(uid_df)
#
#             synth_l.append(synth_df)
#
#         synth_df = pd.concat(synth_l).reset_index(drop=True)
#         synth_df['unique_id'] = synth_df['unique_id'].apply(lambda x: f'Synth_{x}')
#
#         return synth_df
#
#     @abstractmethod
#     def _create_synthetic_ts(self, df: pd.DataFrame):
#         raise NotImplementedError

#
# class TSDataGeneratorTensor(BaseTSGenerator):
#
#     def __init__(self, augment: bool):
#         super().__init__()
#
#         self.augment = augment
#
#     def fit(self, **kwargs):
#         raise NotImplementedError
#
#     def transform(self, tsr: torch.tensor):
#         tsr0, tsr_ = deepcopy(tsr), deepcopy(tsr)
#
#         for i, ts_i in enumerate(tsr_):
#             tsr_[i] = self._transform(ts_i)
#
#         if self.augment:
#             tsr_ = torch.concat([tsr0, tsr_])
#
#         return tsr_
#
#     def _transform(self, tsr: torch.tensor) -> torch.tensor:
#         dtype_ = tsr[0, :].dtype
#
#         arr = tsr[0, :][tsr[1, :] > 0]
#
#         try:
#             synth_arr = self._create_synthetic_ts(arr)
#         except ValueError:
#             synth_arr = arr
#
#         synth_tsr = torch.tensor(synth_arr, dtype=dtype_)
#
#         tsr[0, :][tsr[1, :] > 0] = synth_tsr
#
#         return tsr
#
#     @abstractmethod
#     def _create_synthetic_ts(self, ts: torch.tensor) -> torch.tensor:
#         raise NotImplementedError
