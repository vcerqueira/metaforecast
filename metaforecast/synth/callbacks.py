import copy

import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl


class OntheFlyDataAugmentationCallback(pl.Callback):
    def __init__(self, generator):
        super().__init__()

        self.generator = generator

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        temporal = batch['temporal']

        df_ = self.temporal_to_df(temporal)

        df_synth = self.generator.transform(df_)

        df_aug = pd.concat([df_, df_synth])

        temporal_aug = self.df_to_tensor(df_aug)

        batch['temporal'] = temporal_aug

        return batch

    @staticmethod
    def temporal_to_df(temporal: torch.Tensor) -> pd.DataFrame:
        temporal_ = copy.deepcopy(temporal)

        arr_list = []
        for i, arr in enumerate(temporal_):
            arr_t = arr.numpy().T
            arr_df = pd.DataFrame(arr_t).copy()
            arr_df.columns = ['y', 'y_mask']
            arr_df['ds'] = np.arange(arr_df.shape[0])
            arr_df['unique_id'] = f'ID{i}'

            arr_list.append(arr_df)

        df = pd.concat(arr_list)
        df = df.query('y_mask>0').drop(columns=['y_mask']).reset_index(drop=True)

        return df

    @staticmethod
    def create_mask(df):
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
    def df_to_tensor(cls, df):
        df_ = cls.create_mask(df).copy()

        arr_list = []
        for _, uid_df in df_.groupby('unique_id'):
            arr_list.append(uid_df[['y', 'y_mask']].values.T)

        arr = np.stack(arr_list, axis=0)

        t = torch.tensor(arr, dtype=torch.float32)

        return t
