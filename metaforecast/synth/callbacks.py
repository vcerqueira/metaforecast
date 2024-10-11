import copy

import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd
from neuralforecast.core import TimeSeriesDataset


class OntheFlyDataAugmentationCallback(pl.Callback):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        temporal = batch['temporal']

        print(temporal.keys())
        # print(temporal[:2])

        # return manipulated_batch

    @staticmethod
    def temporal_to_df(temporal: torch.Tensor) -> pd.DataFrame:
        temporal_ = copy.deepcopy(temporal)

        arr_list = []
        for i, arr in enumerate(temporal_):
            arr_t = arr.numpy().T
            arr_df = pd.DataFrame(arr_t).copy()
            arr_df.columns = ['y', 'y_mask']
            arr_df['ds'] = np.arange(arr_df.shape[0])
            arr_df['unique_id'] = i

            arr_list.append(arr_df)

        df = pd.concat(arr_list)
        df = df.query('y_mask>0').drop(columns=['y_mask']).reset_index(drop=True)

        return df

    @staticmethod
    def df_to_temporal(df: pd.DataFrame) -> torch.Tensor:
        temporal_, *_ = TimeSeriesDataset.from_df(df=df)

        return temporal_
