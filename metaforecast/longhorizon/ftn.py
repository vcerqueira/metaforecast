import copy
from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.neighbors import KNeighborsRegressor as KNN
from mlforecast import MLForecast
from mlforecast.target_transforms import Differences
from datasetsforecast.evaluation import accuracy
from datasetsforecast.losses import smape


class ForecastTrajectoryNeighbors(ABC):
    """ ForecastTrajectoryNeighbors

    Forecasted Trajectory Neighbors

    Improving Multi-step Forecasts with Neighbors Adjustments, especially for long horizons

    References:
        Cerqueira, Vitor, Luis Torgo, and Gianluca Bontempi. "Instance-based meta-learning for
        conditionally dependent univariate multi-step forecasting." International Journal of Forecasting (2024).

    """
    KNN_WEIGHTING = 'uniform'
    EVAL_BASE_COLS = ['unique_id', 'ds', 'y', 'horizon']

    def __init__(self,
                 n_neighbors: int,
                 horizon: int,
                 apply_ewm: bool = False,
                 apply_weighting: bool = False,
                 apply_global: bool = False,
                 ewm_smooth: float = 0.6):
        self.base_knn = KNN(n_neighbors=n_neighbors, weights=self.KNN_WEIGHTING)
        self.model = {}

        self.n_neighbors = n_neighbors
        self.horizon = horizon
        self.ewm_smooth = ewm_smooth
        self.apply_ewm = apply_ewm
        self.apply_weighting = apply_weighting
        self.apply_global = apply_global
        self.alpha_weights = {}

        self.uid_insample_traj = {}
        self.model_names = None

    @abstractmethod
    def fit(self, df: pd.DataFrame):
        """ fit

        Fitting in this case means indexation of training data
        """

        raise NotImplementedError

    @abstractmethod
    def predict(self, fcst: pd.DataFrame):
        """ predict

        Correct the forecasts of a model

        :param fcst: predictions in a nixtla-based structure with multi-step forecasts
        :type fcst: pd.DataFrame
        """
        raise NotImplementedError

    def set_alpha_weights(self, alpha: Dict[str, np.ndarray]):
        """ set_alpha_weights

        When weighting the corrected (FTN) forecasts with the original ones you need to set the weights of FTN
        using this function.
        The function expects an array of weights with size equal to the forecasting horizon.
        Each weight should be in a 0-1 range, where 1 means that the final prediction only considers FTN

        :param alpha: the FTN weights for a dict of forecasting models
        :type alpha: dict, with keys being the model names (str) and the values a numpy array with weight values
        for each horizon.

        :return: self
        """
        for k in alpha:
            assert len(alpha[k]) == self.horizon

        self.alpha_weights = alpha

    def _smooth_series(self, df: pd.DataFrame):
        """ _smooth_series

        Apply an exponential moving average to a time series dataset

        :param df: time series dataset, following a nixtla-based structure, with unique_id, ds, y
        :type df: pd.DataFrame

        :return: smoothed df
        """
        smoothed_uid_l = []
        for uid, uid_df in df.groupby(['unique_id']):
            uid_df['y'] = uid_df['y'].ewm(alpha=self.ewm_smooth).mean()

            smoothed_uid_l.append(uid_df)

        smooth_df = pd.concat(smoothed_uid_l)

        return smooth_df

    def reset_learning(self):
        """ reset_learning

        Reset previous meta-models
        """
        self.model = {}
        self.uid_insample_traj = {}


class MLForecastFTN(ForecastTrajectoryNeighbors):
    BASE_PARAMS = {'models': [], 'freq': ''}
    METADATA = ['unique_id', 'ds']
    GLB_UID = 'glob'

    def __init__(self,
                 n_neighbors: int,
                 horizon: int,
                 apply_ewm: bool = False,
                 apply_weighting: bool = False,
                 apply_diff1: bool = False,
                 apply_global: bool = False,
                 ewm_smooth: float = 0.6):

        super().__init__(n_neighbors=n_neighbors,
                         horizon=horizon,
                         apply_ewm=apply_ewm,
                         apply_weighting=apply_weighting,
                         apply_global=apply_global,
                         ewm_smooth=ewm_smooth)

        self.apply_diff1 = apply_diff1
        self.lags = range(1, self.horizon)

        self.preprocess_params = {'lags': self.lags, **self.BASE_PARAMS}

        if self.apply_diff1:
            self.preprocess_params['target_transforms'] = [Differences([1])]

        self.preprocess = MLForecast(**self.preprocess_params)

    def fit(self, df: pd.DataFrame):
        """ fit
        Fitting in this case means indexation of training data
        """

        self.reset_learning()

        if self.apply_ewm:
            df = self._smooth_series(df)

        rec_df = self.preprocess.preprocess(df)
        rec_df = rec_df.rename(columns={'y': 'lag0'})

        lag_names = [f'lag{i}' for i in range(self.horizon)]

        if self.apply_global:
            self.uid_insample_traj[self.GLB_UID] = rec_df[lag_names[::-1]].values
            self.model[self.GLB_UID] = copy.deepcopy(self.base_knn)
            self.model[self.GLB_UID] = self.model[self.GLB_UID].fit(self.uid_insample_traj[self.GLB_UID],
                                                                    self.uid_insample_traj[self.GLB_UID][:, 0])
        else:
            for uid, df_ in tqdm(rec_df.groupby('unique_id')):
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
        self.model_names = [x for x in fcst.columns if x not in self.METADATA]

        fcst_ftn = []
        for uid, df_ in tqdm(fcst.groupby('unique_id')):
            ftn_df = df_.copy()
            for m in self.model_names:
                fcst_ = df_[m].values
                x0 = fcst_[0]

                if self.apply_global:
                    _, k_neighbors = self.model[self.GLB_UID].kneighbors(fcst_.reshape(1, -1))
                    ftn_knn = self.uid_insample_traj[self.GLB_UID][k_neighbors[0], :]
                else:
                    _, k_neighbors = self.model[uid].kneighbors(fcst_.reshape(1, -1))
                    ftn_knn = self.uid_insample_traj[uid][k_neighbors[0], :]

                ftn_fcst = ftn_knn.mean(axis=0)

                if self.apply_diff1:
                    ftn_fcst = np.r_[x0, ftn_fcst[1:]].cumsum()

                name_ = f'{m}(FTN)'
                if self.apply_weighting:
                    if m in self.alpha_weights:
                        w = self.alpha_weights[m]

                        ftn_fcst = ftn_fcst * w + fcst_ * (1 - w)
                        name_ = f'{m}(WFTN)'

                ftn_df[name_] = ftn_fcst

            fcst_ftn.append(ftn_df)

        fcst_ftn_df = pd.concat(fcst_ftn)

        return fcst_ftn_df

    def alpha_cv_scoring(self,
                         cv: pd.DataFrame,
                         model_names: Optional[List[str]] = None):

        if model_names is None:
            assert self.model_names is not None
            models = self.model_names
        else:
            models = model_names

        weights = {}
        for m in models:
            cv_ = cv[self.EVAL_BASE_COLS + [m, f'{m}(FTN)']]

            eval_by_horizon = {}
            for h_, h_df in cv_.groupby('horizon'):
                h_df = accuracy(h_df, [smape], agg_by=['unique_id'])
                h_df_avg = h_df.drop(columns=['horizon', 'metric', 'unique_id']).mean()

                eval_by_horizon[h_] = h_df_avg

            h_eval_df = pd.DataFrame(eval_by_horizon).T

            horizon_weights = h_eval_df.apply(lambda x: 1 - softmax(x)[1], axis=1)
            weights[m] = horizon_weights.values

        return weights
