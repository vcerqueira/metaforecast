from abc import ABC, abstractmethod
import copy

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor as KNN
from mlforecast import MLForecast
from mlforecast.target_transforms import Differences


# from metaforecast.utils.barycenters import BarycentricAveraging


class ForecastTrajectoryNeighbors(ABC):
    """
    Forecasted Trajectory Neighbors

    Improving Multi-step Forecasts with Neighbors Adjustments
    """
    KNN_WEIGHTING = 'uniform'

    def __init__(self,
                 n_neighbors: int,
                 horizon: int,
                 apply_ewm: bool = False,
                 apply_partial: bool = False,
                 ewm_smooth: float = 0.6):
        self.base_knn = KNN(n_neighbors=n_neighbors, weights=self.KNN_WEIGHTING)
        self.model = {}

        self.n_neighbors = n_neighbors
        self.horizon = horizon
        self.ewm_smooth = ewm_smooth
        self.apply_ewm = apply_ewm
        self.apply_partial = apply_partial
        self.alpha_w = np.linspace(start=0, stop=1, num=self.horizon)

        self.uid_insample_traj = {}

    @abstractmethod
    def fit(self, df: pd.DataFrame):
        """

        Fitting in this case means indexation of training data
        """

        raise NotImplementedError

    @abstractmethod
    def predict(self, fcst: pd.DataFrame):
        """
        Making predictions

        Y_hat: pd.DF with the multi-step forecasts of a base-model
        Y_hat has shape (n_test_observations, forecasting_horizon)
        """

        raise NotImplementedError

    def set_alpha_weights(self, alpha: np.ndarray):
        assert len(alpha) == self.horizon

        self.alpha_w = alpha

    def _smooth_series(self, df: pd.DataFrame):
        smoothed_uid_l = []
        for uid, uid_df in df.groupby(['unique_id']):
            uid_df['y'] = uid_df['y'].ewm(alpha=self.ewm_smooth).mean()

            smoothed_uid_l.append(uid_df)

        smooth_df = pd.concat(smoothed_uid_l)

        return smooth_df


class MLForecastFTN(ForecastTrajectoryNeighbors):
    BASE_PARAMS = {'models': [], 'freq': ''}

    def __init__(self,
                 n_neighbors: int,
                 horizon: int,
                 apply_ewm: bool = False,
                 apply_partial: bool = False,
                 apply_1_diff: bool = False,
                 # barycenter: str = 'euclidean',
                 ewm_smooth: float = 0.6):

        super().__init__(n_neighbors=n_neighbors,
                         horizon=horizon,
                         apply_ewm=apply_ewm,
                         apply_partial=apply_partial,
                         ewm_smooth=ewm_smooth)

        self.apply_1_diff = apply_1_diff
        self.lags = range(1, self.horizon)

        self.preprocess_params = {'lags': self.lags, **self.BASE_PARAMS}
        # self.barycenter = barycenter

        if self.apply_1_diff:
            self.preprocess_params['target_transforms'] = [Differences([1])]

        self.preprocess = MLForecast(**self.preprocess_params)

    def fit(self, df: pd.DataFrame):
        """

        Fitting in this case means indexation of training data
        """

        if self.apply_ewm:
            df = self._smooth_series(df)

        rec_df = self.preprocess.preprocess(df)
        rec_df = rec_df.rename(columns={'y': 'lag0'})

        lag_names = [f'lag{i}' for i in range(self.horizon)]

        for uid, df_ in rec_df.groupby('unique_id'):
            self.uid_insample_traj[uid] = df_[lag_names[::-1]].values
            self.model[uid] = copy.deepcopy(self.base_knn)
            self.model[uid] = self.model[uid].fit(self.uid_insample_traj[uid],
                                                  self.uid_insample_traj[uid][:, 0])

    def predict(self, fcst: pd.DataFrame):
        """
        Making predictions

        Y_hat: pd.DF with the multi-step forecasts of a base-model
        Y_hat has shape (n_test_observations, forecasting_horizon)
        """
        model_names = [x for x in fcst.columns if x not in ['unique_id', 'ds']]

        fcst_ftn = []
        for uid, df_ in fcst.groupby('unique_id'):
            ftn_df = df_.copy()
            for m in model_names:
                # m='lgbm'
                fcst_ = df_[m].values
                x0 = fcst_[0]

                _, k_neighbors = self.model[uid].kneighbors(fcst_.reshape(1, -1))

                ftn_knn = self.uid_insample_traj[uid][k_neighbors[0], :]

                ftn_fcst = ftn_knn.mean(axis=0)
                # ftn_fcst = BarycentricAveraging.calc_average(ftn_knn, self.barycenter)

                if self.apply_1_diff:
                    ftn_fcst = np.r_[x0, ftn_fcst[1:]].cumsum()

                if self.apply_partial:
                    ftn_fcst = ftn_fcst * self.alpha_w + fcst_ * (1 - self.alpha_w)

                ftn_df[m] = ftn_fcst

            fcst_ftn.append(ftn_df)

        fcst_ftn_df = pd.concat(fcst_ftn)

        return fcst_ftn_df
