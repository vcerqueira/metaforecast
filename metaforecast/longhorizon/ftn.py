import copy
from warnings import simplefilter
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

simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


class ForecastTrajectoryNeighbors(ABC):
    """ ForecastTrajectoryNeighbors

    Forecasted Trajectory Neighbors

    Improving Multi-step Forecasts with Neighbors Adjustments, especially for long horizons

    References:
        Cerqueira, Vitor, Luis Torgo, and Gianluca Bontempi. "Instance-based meta-learning for
        conditionally dependent univariate multistep forecasting."
        International Journal of Forecasting (2024).

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

        :param fcst: predictions in a nixtla-based structure with multistep forecasts
        :type fcst: pd.DataFrame
        """
        raise NotImplementedError

    def set_alpha_weights(self, alpha: Dict[str, np.ndarray]):
        """ set_alpha_weights

        When weighting the corrected (FTN) forecasts with the original ones you need to set the
        weights of FTN using this function.
        The function expects an array of weights with size equal to the forecasting horizon.
        Each weight should be in a 0-1 range, where 1 means that the final prediction only
        considers FTN

        :param alpha: the FTN weights for a dict of forecasting models
        :type alpha: dict, with keys being the model names (str) and the values a numpy array
        with weight values for each horizon.

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

    @staticmethod
    def get_horizon(cv: pd.DataFrame):
        """ get_horizon

        Get the forecasting horizon for cross-validation results.

        :param cv: cross-validation results from a nixtla-based pipeline
        :type cv: pd.DataFrame with nixtla-based columns: unique_id, cutoff, ds

        :return: cv with horizon column
        """
        if 'horizon' in cv.columns:
            raise ValueError('"horizon" column already in the dataset.')

        if 'cutoff' in cv.columns:
            cv_g = cv.groupby(['unique_id', 'cutoff'])
        else:
            cv_g = cv.groupby(['unique_id'])

        horizon = []
        for g, df in cv_g:
            df = df.sort_values('ds')
            h = np.asarray(range(1, df.shape[0] + 1))
            hs = {
                'horizon': h,
                'ds': df['ds'].values,
                'unique_id': df['unique_id'].values,
            }
            if 'cutoff' in df.columns:
                hs['cutoff'] = df['cutoff'].values,

            hs = pd.DataFrame(hs)
            horizon.append(hs)

        horizon = pd.concat(horizon)

        if 'cutoff' in cv.columns:
            cv = cv.merge(horizon, on=['unique_id', 'ds', 'cutoff'])
        else:
            cv = cv.merge(horizon, on=['unique_id', 'ds'])

        return cv


class MLForecastFTN(ForecastTrajectoryNeighbors):
    """ MLForecastFTN

    FTN based on mlforecast ecosystem

    FTN (Forecasted Trajectory Neighbors) is an instance-based (good old KNN) approach for improving
    multistep forecasts, especially for long horizons.

    References:
        Cerqueira, Vitor, Luis Torgo, and Gianluca Bontempi. "Instance-based meta-learning for
        conditionally dependent univariate multistep forecasting."
        International Journal of Forecasting (2024).

    Example usage (CHECK NOTEBOOKS FOR MORE SERIOUS EXAMPLES):
    >>> import lightgbm as lgb
    >>> from mlforecast import MLForecast
    >>> from datasetsforecast.m3 import M3
    >>>
    >>> from metaforecast.longhorizon import MLForecastFTN as FTN
    >>>
    >>> # loading data
    >>> df, *_ = M3.load('.', group='Monthly')
    >>> horizon = 18
    >>> # setting up forecasting model
    >>> models = {'lgbm': lgb.LGBMRegressor(verbosity=-1), }
    >>>
    >>> mlf = MLForecast(
    >>>     models=models,
    >>>     freq='ME',
    >>>     lags=range(1, 13),
    >>> )
    >>>
    >>> # setting up FTN
    >>> ftn = FTN(horizon=horizon,
    >>>           n_neighbors=30,
    >>>           apply_diff1=True)
    >>>
    >>> mlf.fit(df=df)
    >>> ftn.fit(df)
    >>>
    >>> fcst_mlf = mlf.predict(h=horizon)
    >>> fcst_ftn = ftn.predict(fcst_mlf)
    """

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
        """
        :param n_neighbors: Number of nearest neighbors (training subsequences) used in KNN.
        About 100 neighbors tends to work well for high frequency series. See notebooks
        :type n_neighbors: int

        :param horizon: Forecasting horizon
        :type horizon: int

        :param apply_ewm: Whether to apply an exponential moving average to smooth the
        time series before fitting the KNN
        :param apply_ewm: bool

        :param apply_weighting: Whether to weight the FTN predictions with the original forecasts.
        The weights of FTN should be set using the set_alpha_weights method.
        See notebooks for an example of how to set these using cross-validation
        :type apply_weighting: bool

        :param apply_diff1: Whether to apply first differences before fitting the KNN to
        stabilize the mean level
        :type apply_diff1: bool

        :param apply_global: Whether to fit a KNN for all dataset (True) or for each time series
        (unique_id) (False) A global approach tends to perform poorly (wip)
        :type apply_global: bool

        :param ewm_smooth: Exponential moving average strength parameter. Defaults to 0.6.
        See https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html
        :type ewm_smooth: float
        """

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

        Fitting FTN.
        Fitting in this case means indexation of training data

        :param df: Time series dataset following a Nixtla structure (unique_id, ds, y)
        :type df: pd.DataFrame
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
            self.model[self.GLB_UID] = \
                self.model[self.GLB_UID].fit(self.uid_insample_traj[self.GLB_UID],
                                             self.uid_insample_traj[self.GLB_UID][:, 0])
        else:
            for uid, df_ in tqdm(rec_df.groupby('unique_id')):
                self.uid_insample_traj[uid] = df_[lag_names[::-1]].values
                self.model[uid] = copy.deepcopy(self.base_knn)
                self.model[uid] = self.model[uid].fit(self.uid_insample_traj[uid],
                                                      self.uid_insample_traj[uid][:, 0])

    def predict(self, fcst: pd.DataFrame):
        """ predict

        Correcting the forecasts of a model using KNN

        :param fcst: predictions in a nixtla-based structure with multistep forecasts
        :type fcst: pd.DataFrame
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
        """ alpha_cv_scoring

        Computing the best FTN weights for each horizon.
        See notebooks for an example.

        :param cv: Validation or cross-validation results from a nixtla-based procedure
        :type cv: pd.DataFrame

        :param model_names: List of model(s) to estimate the weights for.
        If None, uses the models found during fit
        :type model_names: Optional list of str
        """

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
