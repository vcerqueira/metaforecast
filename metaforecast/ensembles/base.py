from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd
from neuralforecast.losses.numpy import smape

from metaforecast.ensembles.expert_loss import (
    AbsoluteLoss,
    LogLoss,
    PercentageLoss,
    PinballLoss,
    SquaredLoss,
)
from metaforecast.utils.normalization import Normalizations

EXPERT_LOSS = {
    "square": SquaredLoss,
    "pinball": PinballLoss,
    "percentage": PercentageLoss,
    "absolute": AbsoluteLoss,
    "log": LogLoss,
}


class ForecastingEnsemble(ABC):
    """ForecastingEnsemble

    Abstract class for a forecasting ensemble method
    """

    METADATA = ["unique_id", "ds", "y"]
    METADATA_NO_T = ["unique_id", "ds"]

    WINDOW_SIZE_BY_FREQ = {
        "H": 48,
        "D": 14,
        "W": 16,
        "M": 12,
        "ME": 12,
        "MS": 12,
        "Q": 4,
        "QS": 4,
        "Y": 6,
        "": -1,
    }

    def __init__(self):
        super().__init__()

        self.models = []
        self.model_names = None
        self.tot_n_models = -1
        self.n_models = -1
        self.n_poor_models = -1
        self.trim_ratio = 0
        self.window_size = 0

    @abstractmethod
    def fit(self, **kwargs):
        """fit

        Fits the ensemble combination rule

        Parameters
        ----------
        kwargs: Not defined
            Whatever input value the ensemble takes.

        Returns
        -------
        ForecastingEnsemble
            self, optional

        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, **kwargs):
        """predict

        Predicts the weights of ensemble combination rule

        Parameters
        ----------
        kwargs: Not defined
            Whatever input value the ensemble takes.

        Returns
        -------
        Array-like
            Weights of each model in the ensemble

        """
        raise NotImplementedError

    def update_weights(self, fcst: pd.DataFrame):
        """Update model performance statistics for dynamic ensemble selection.

        Parameters
        ----------
        fcst: pd.DataFrame
            Dataset containing actual values and model predictions.
            Expected columns:
            - unique_id: Series identifier
            - ds: Timestamp
            - model_name: Predictions of model with name "model_name"
            - y: Actual values

        """
        raise NotImplementedError

    def evaluate_base_fcst(
        self, insample_fcst: pd.DataFrame, use_window: bool
    ) -> pd.DataFrame:
        """Evaluate ensemble members' accuracy by series using SMAPE.

        Computes Symmetric Mean Absolute Percentage Error (SMAPE) for each base model
        on each individual time series. Can evaluate using either full history or
        recent window of observations.

        Parameters
        ----------
        insample_fcst : pd.DataFrame
            In-sample or cross-validation predictions and actual values.
            Expected columns:
                - unique_id: Series identifier
                - ds: Timestamp
                - model_name: Predictions of model with name "model_name"
                - y: Actual values

        use_window : bool, default=False
            If True, evaluate only the last self.window_size observations
            If False, evaluate using complete history

        Returns
        -------
        pd.DataFrame
            Performance metrics for each model-series combination.

        """

        all_scores, window_scores = {}, {}
        in_sample_loss_g = insample_fcst.groupby("unique_id")
        for uid, uid_df in in_sample_loss_g:
            uid_a_loss, uid_w_loss = {}, {}
            for m in self.model_names:
                uid_a_loss[m] = smape(y=uid_df["y"], y_hat=uid_df[m])
                try:
                    uid_w_loss[m] = smape(
                        y=uid_df.tail(self.window_size)["y"],
                        y_hat=uid_df.tail(self.window_size)[m],
                    )
                except AssertionError:
                    uid_w_loss[m] = np.nan

            all_scores[uid] = uid_a_loss
            window_scores[uid] = uid_w_loss

        all_scr_df = pd.DataFrame(all_scores).T
        wdw_scr_df = pd.DataFrame(window_scores).T

        if use_window:
            return wdw_scr_df

        return all_scr_df

    @abstractmethod
    def _weights_by_uid(self, **kwargs):
        raise NotImplementedError

    def _set_n_models(self):
        """_set_n_models

        Setting the number of models to be used based on the trim ratio
        """
        self.tot_n_models = len(self.model_names)

        self.n_models = int(self.trim_ratio * self.tot_n_models)
        self.n_models = max(self.n_models, 1)

        self.n_poor_models = self.tot_n_models - self.n_models

    def _get_top_k(self, scores: pd.Series) -> List[str]:
        """_get_top_k

        Get the top k models based on loss scores

        :param scores: (pd.Series) models error scores (to minimize)

        :return List[str] of the top k models
        """
        return scores.sort_values().index.tolist()[: self.n_models]

    @staticmethod
    def _weights_from_errors(scores: pd.Series) -> pd.Series:
        """_weights_from_errors

        Transforming error scores into convex weights

        :param scores: (pd.Series) error scores of each ensemble member

        :return: (pd.Series) ensemble member weights
        """

        weights_ = Normalizations.normalize_and_proportion(-scores)

        return weights_

    @staticmethod
    def _weighted_average(pred: pd.Series, weights: pd.DataFrame):
        """_weighted_average

        Compute a weighted average of a prediction based on models' weights

        :param pred: forecast
        :param weights: weights

        :return: ensemble weighted forecast
        """

        w = weights.loc[pred["unique_id"]]

        wa = (pred[w.index] * w).sum()

        return wa

    @staticmethod
    def _assert_fcst(fcst: pd.DataFrame):
        assert (
            "unique_id" in fcst.columns
        ), '"unique_id" should be included in the predictions object'


class Mixture(ForecastingEnsemble):
    """Online learning ensemble that combines forecasts using regret minimization.

    Adapts model weights over time by minimizing prediction regret. Supports
    multiple loss functions and can optimize weights globally or per series.

    Parameters
    ----------
    loss_type : {'square', 'absolute', 'percentage', 'log', 'pinball'}
    Loss function for computing model weights:
        - square: Mean squared error
        - absolute: Mean absolute error
        - percentage: Mean absolute percentage error
        - log: Log loss
        - pinball: Quantile loss

    gradient : bool, default=False
        If True, use gradient for weight updates

    trim_ratio : float, default=1.0
        Proportion of models to retain in ensemble, between 0 and 1.
        Models are selected based on cumulative performance:
        - 1.0 keeps all models
        - 0.5 keeps top 50% of models
        - Lower values create a more selective ensemble

    weight_by_uid : bool, default=False
        If True, compute separate weights for each time series
        If False, use global weights across all series
        Note: Setting to True may be computationally intensive
        for datasets with many series

    Notes
    -----
    This implementation follows the implementation of opera's R package.

    References
    ----------
    Cesa-Bianchi, N., & Lugosi, G. (2006). Prediction, learning, and games.
    """

    def __init__(
        self,
        loss_type: str,
        gradient: bool,
        trim_ratio: float,
        weight_by_uid: bool,
    ):
        self.alias = "Mixture"

        super().__init__()

        assert loss_type in EXPERT_LOSS

        self.gradient = gradient
        self.loss_type = loss_type
        self.trim_ratio = trim_ratio
        self.weight_by_uid = weight_by_uid

        self.eta = None
        self.regret = None
        self.weights = None
        self.ensemble_fcst = None

        self.uid_weights = {}
        self.uid_coefficient = {}

    # pylint: disable=arguments-differ
    def fit(self, insample_fcst: pd.DataFrame):
        """Fitting the dynamic combination rule

        Parameters
        ----------
        insample_fcst : pd.DataFrame
            Forecast and actual values dataset formatted like mlforecast cross-validation output.
            Contains either:
                - In-sample forecasts (predictions on training data)
                - Cross-validation results (out-of-sample predictions)

            Expected columns:
                - unique_id (or other id_col): Identifier for each series
                - ds (or other time_col): Timestamp
                - y (or other target_col): Actual values
                - *model_name*: Predictions by a model with name *model_name*

        Returns
        -------
        self, with computed self.weights

        """

        if self.model_names is None:
            self.model_names = insample_fcst.columns.to_list()
            self.model_names = [
                x for x in self.model_names if x not in self.METADATA + ["h"]
            ]

        self._initialize_params(insample_fcst)
        self._set_n_models()

        if self.weight_by_uid:
            self._fit_by_uid(insample_fcst)
        else:
            self._fit_all(insample_fcst)

    def _fit_by_uid(self, insample_fcst: pd.DataFrame):
        grouped_fcst = insample_fcst.groupby("unique_id")

        for uid, fcst_uid in grouped_fcst:
            y = fcst_uid["y"].values

            fcst_uid = fcst_uid.reset_index(drop=True)
            fcst_uid = fcst_uid.drop(columns=self.METADATA)
            if "h" in fcst_uid.columns:
                fcst_uid = fcst_uid.drop(columns="h")

            self._initialize_params(fcst_uid)

            self._update_mixture(fcst_uid, y)

            self.weights = pd.DataFrame(self.weights, columns=self.model_names)

            self.uid_weights[uid] = self.weights.iloc[-1]
            self.uid_coefficient[uid] = self._weights_from_regret()

    def _fit_all(self, insample_fcst: pd.DataFrame):
        """

        Can comment the initial sorting to have the estimation match the r bridge

        :param insample_fcst: forecasts
        """

        insample_fcst_ = insample_fcst.sort_values("ds")

        uid_list = insample_fcst_["unique_id"].unique().tolist()

        y = insample_fcst_["y"].values

        fcst = insample_fcst_.reset_index(drop=True)
        fcst = fcst.drop(columns=self.METADATA)
        if "h" in fcst.columns:
            fcst = fcst.drop(columns="h")

        self._update_mixture(fcst, y)
        self.weights = pd.DataFrame(self.weights, columns=self.model_names)

        for uid in uid_list:
            self.uid_weights[uid] = self.weights.iloc[-1]
            self.uid_coefficient[uid] = self._weights_from_regret()

    # pylint: disable=arguments-differ
    def predict(self, fcst: pd.DataFrame, **kwargs):
        """Combine ensemble member forecasts using the mixture.

        Parameters
        ----------
        fcst : pd.DataFrame
            Forecasts from individual ensemble members.
            Expected columns: ['unique_id', 'ds', 'model_name1', 'model_name2', ...]

        Returns
        -------
        pd.Series
            Combined ensemble forecasts for h periods ahead.

        """
        self._assert_fcst(fcst)

        weights = pd.DataFrame(self.uid_weights).T
        # weights = pd.DataFrame(self.uid_coefficient).T

        if self.trim_ratio < 1:
            weights = self._weights_by_uid(weights)

        fcst_c = fcst.apply(lambda x: self._weighted_average(x, weights), axis=1)
        fcst_c.name = self.alias

        return fcst_c

    def _calc_loss(self, fcst: pd.Series, y: float, fcst_c: float):
        if self.gradient:
            loss = EXPERT_LOSS[self.loss_type].gradient(fcst=fcst, y=y, fcst_c=fcst_c)
        else:
            loss = EXPERT_LOSS[self.loss_type].loss(fcst=fcst, y=y)

        return loss

    # pylint: disable=arguments-differ
    def update_weights(self, fcst: pd.DataFrame):
        raise NotImplementedError

    def _initialize_params(self, fcst: pd.DataFrame):
        raise NotImplementedError

    def _weights_from_regret(self, **kwargs):
        raise NotImplementedError

    def _update_mixture(self, fcst: pd.DataFrame, y: np.ndarray, **kwargs):
        raise NotImplementedError

    # pylint: disable=arguments-differ
    def _weights_by_uid(self, weights: pd.DataFrame, **kwargs):
        neg_w = -weights

        top_overall = self._get_top_k(-weights.mean())
        top_by_uid = neg_w.apply(self._get_top_k, axis=1)

        uid_weights = {}
        for uid, w in weights.iterrows():
            if self.weight_by_uid:
                poor_models = [x not in top_by_uid[uid] for x in w.index]
            else:
                poor_models = [x not in top_overall for x in w.index]

            w[poor_models] = 0
            w /= w.sum()

            uid_weights[uid] = w

        weights_df = pd.DataFrame(uid_weights).T
        weights_df.sum()
        weights_df.index.name = "unique_id"

        return weights_df

    @staticmethod
    def _calc_ensemble_fcst(fcst: pd.Series, weight: pd.Series):
        # form the mixture and the prediction
        mixture = weight / np.sum(weight)

        fcst_c = np.sum(fcst * mixture)

        return mixture, fcst_c


class BaseADE(ForecastingEnsemble):
    """
    ADE

    Arbitrated Dynamic Ensemble

    """

    def __init__(
        self,
        window_size: int,
        trim_ratio: float,
        trim_by_uid: bool,
        meta_model,
    ):
        super().__init__()

        self.window_size = window_size
        self.trim_ratio = trim_ratio
        self.trim_by_uid = trim_by_uid
        self.meta_model = meta_model

        self.alias = "ADE"

    def fit(self, **kwargs):
        raise NotImplementedError

    def predict(self, **kwargs):
        raise NotImplementedError

    # pylint: disable=arguments-differ
    def update_weights(self, **kwargs):
        raise NotImplementedError

    def _weights_by_uid(self, **kwargs):
        raise NotImplementedError
