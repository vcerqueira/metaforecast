import copy
from typing import Union

import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl

from metaforecast.synth.generators._base import (PureSyntheticGenerator,
                                                 SemiSyntheticGenerator,
                                                 SemiSyntheticTransformer)

TSGenerator = Union[PureSyntheticGenerator, SemiSyntheticGenerator, SemiSyntheticTransformer]


class OnlineDataAugmentationCallback(pl.Callback):
    """ OnlineDataAugmentationCallback

    Online (batch-by-batch) data augmentation via a callback

    Example usage
    >>> from datasetsforecast.m3 import M3
    >>> from neuralforecast import NeuralForecast
    >>> from neuralforecast.models import NHITS
    >>>
    >>> from metaforecast.utils.data import DataUtils
    >>> from metaforecast.synth import SeasonalMBB
    >>> from metaforecast.synth.callbacks import OnlineDataAugmentationCallback
    >>>
    >>> augmentation_cb = OnlineDataAugmentationCallback(generator=SeasonalMBB(seas_period=12))
    >>>
    >>> df, *_ = M3.load('.', group='Monthly')
    >>>
    >>> horizon = 24
    >>>
    >>> train, test = DataUtils.train_test_split(df, horizon)
    >>>
    >>> models = [NHITS(input_size=horizon,
    >>>                 h=horizon,
    >>>                 start_padding_enabled=True,
    >>>                 callbacks=[augmentation_cb])]
    >>>
    >>> nf = NeuralForecast(models=models, freq='M')
    >>>
    >>> nf.fit(df=train)
    >>>
    >>> fcst = nf.predict()

    """

    def __init__(self, generator):
        """
        :param generator: A synthetic time series generator
        :type generator: An object of that extends BaseTimeSeriesGenerator, i.e. PureSyntheticGenerator,
        SemiSyntheticGenerator, or SemiSyntheticTransformer

        """
        super().__init__()

        self.generator = generator

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """
        Applying data augmentation after getting a batch of time series for training
        """
        temporal = batch['temporal']

        df_ = self.temporal_to_df(temporal)

        df_synth = self.generator.transform(df_)

        df_aug = pd.concat([df_, df_synth])

        temporal_aug = self.df_to_tensor(df_aug)

        if isinstance(temporal, torch.mps.Tensor):
            temporal_aug = temporal_aug.to('mps')

        batch['temporal'] = temporal_aug

        return batch

    @staticmethod
    def temporal_to_df(temporal: torch.Tensor) -> pd.DataFrame:
        """ temporal_to_df

        Converting the batch of time series into a DataFrame structure with (unique_id, ds, y) data.

        :param temporal: A tensor with a batch of time series
        :type temporal: torch.Tensor

        :return: Time series dataset as a pd.DataFrame
        """
        temporal_ = copy.deepcopy(temporal)

        arr_list = []
        for i, arr in enumerate(temporal_):
            if isinstance(arr, torch.mps.Tensor):
                arr = arr.cpu()

            arr_t = arr.cpu().numpy().T

            arr_df = pd.DataFrame(arr_t).copy()
            arr_df.columns = ['y', 'y_mask']
            arr_df['ds'] = np.arange(arr_df.shape[0])
            arr_df['unique_id'] = f'ID{i}'

            arr_list.append(arr_df)

        df = pd.concat(arr_list)
        df = df.query('y_mask>0').drop(columns=['y_mask']).reset_index(drop=True)

        return df

    @staticmethod
    def create_mask(df) -> pd.DataFrame:
        """ create_mask

        Transforming the time series into the same size (equal to the largest one) using a mask.
        Shorter time series will be left-padded with 0 and the padded samples will be denoted by a binary variable.

        :param df: Time series dataset as pd.DataFrame with columns (unique_id, ds, y)
        :type df: pd.DataFrame

        :return: Masked time series dataset
        """
        uids = df['unique_id'].unique()
        all_ds = np.arange(0, df['ds'].max() + 1)

        df_extended = pd.DataFrame([(uid, ds)
                                    for uid in uids
                                    for ds in all_ds],
                                   columns=['unique_id', 'ds'])

        result = pd.merge(df_extended, df, on=['unique_id', 'ds'], how='left')

        result['y_mask'] = (~result['y'].isna()).astype(int)
        result['y'] = result['y'].fillna(0)

        result = result.sort_values(['unique_id', 'ds'])
        result = result.reset_index(drop=True)

        return result

    @classmethod
    def df_to_tensor(cls, df: pd.DataFrame) -> torch.Tensor:
        """ df_to_tensor

        Converting a time series dataset from pd.DataFrame into a torch.Tensor based on neuralforecast's workflow

        :param df: Masked time series dataset with (unique_id, ds, y, y_mask) variables

        :return: Time series dataset as a torch Tensor
        """

        df_ = cls.create_mask(df).copy()

        arr_list = []
        for _, uid_df in df_.groupby('unique_id'):
            arr_list.append(uid_df[['y', 'y_mask']].values.T)

        arr = np.stack(arr_list, axis=0)

        t = torch.tensor(arr, dtype=torch.float32)

        return t
