import numpy as np
import pandas as pd

from tslearn.barycenters import dtw_barycenter_averaging_subgradient as dtw

from metaforecast.synth.generators._base import SemiSyntheticGenerator


class DBA(SemiSyntheticGenerator):
    DTW_PARAMS = {'max_iter': 5, 'tol': 1e-3}

    def __init__(self, max_n_uids: int, dirichlet_alpha: float = 1.0):
        super().__init__(alias='DBA')

        self.max_n_uids = max_n_uids
        self.dirichlet_alpha = dirichlet_alpha

    def transform(self, df: pd.DataFrame, n_series: int = -1):
        unq_uids = df['unique_id'].unique()

        if n_series < 0:
            n_series = len(unq_uids)

        dataset = []
        for _ in range(n_series):
            n_uids = np.random.randint(1, self.max_n_uids + 1)

            selected_uids = np.random.choice(unq_uids, n_uids, replace=False).tolist()

            df_uids = df.query('unique_id == @selected_uids')

            ts_df = self._create_synthetic_ts(df_uids)
            ts_df['unique_id'] = f'{self.alias}_{self.counter}'
            self.counter += 1

            dataset.append(ts_df)

        synth_df = pd.concat(dataset).reset_index(drop=True)

        return synth_df

    def _create_synthetic_ts(self, df: pd.DataFrame) -> pd.DataFrame:
        y_list = [y['y'].values for _, y in df.groupby('unique_id')]
        uid_size = df['unique_id'].value_counts()

        ds = df.query(f'unique_id=="{uid_size.index[0]}"')['ds'].values

        w = self.sample_weights_dirichlet(1, len(y_list))

        synth_y = dtw(X=y_list, weights=w, **self.DTW_PARAMS)
        synth_y = synth_y.flatten()

        synth_df = pd.DataFrame({'ds': ds[:len(synth_y)], 'y': synth_y})

        return synth_df
