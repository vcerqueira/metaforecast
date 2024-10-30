from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class LossFunctions(ABC):
    """LossFunctions

    Abstract class for loss functions and respective gradient.

    These loss functions are used in the context of weighting expert advice, i.e. ensemble learning

    """

    @staticmethod
    @abstractmethod
    def loss(fcst: pd.Series, y: float):
        """
        :param fcst: pd.Series with predictions from set of experts
        :param y: float with actual value
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def gradient(fcst: pd.Series, y: float, fcst_c: float):
        """
        Gradient Loss of a set of experts

        :param fcst: pd.Series with predictions from set of experts
        :param y: float with actual value
        :param fcst_c: float with combined prediction
        """
        raise NotImplementedError


class SquaredLoss(LossFunctions):
    @staticmethod
    def loss(fcst: pd.Series, y: float) -> pd.Series:
        return (fcst - y) ** 2

    @staticmethod
    def gradient(fcst: pd.Series, y: float, fcst_c: float) -> pd.Series:
        return 2 * (fcst_c - y) * fcst


class AbsoluteLoss(LossFunctions):
    @staticmethod
    def loss(fcst: pd.Series, y: float) -> pd.Series:
        return np.abs(fcst - y)

    @staticmethod
    def gradient(fcst: pd.Series, y: float, fcst_c: float) -> pd.Series:
        return np.sign(fcst_c - y) * fcst


class PercentageLoss(LossFunctions):
    @staticmethod
    def loss(fcst: pd.Series, y: float) -> pd.Series:
        return (np.abs(fcst - y)) / y

    @staticmethod
    def gradient(fcst: pd.Series, y: float, fcst_c: float) -> pd.Series:
        return np.sign(fcst_c - y) * (fcst / y)


class LogLoss(LossFunctions):
    @staticmethod
    def loss(fcst: pd.Series, y: float) -> pd.Series:
        return -np.log(fcst)

    @staticmethod
    def gradient(fcst: pd.Series, y: float, fcst_c: float) -> pd.Series:
        return -(fcst / fcst_c)


class PinballLoss(LossFunctions):
    @staticmethod
    def loss(fcst: pd.Series, y: float, tau: float = 0.5) -> pd.Series:
        sign_loss = (y < fcst).astype(int)

        loss = (sign_loss - tau) * (fcst - y)
        return loss

    @staticmethod
    def gradient(
        fcst: pd.Series, y: float, fcst_c: float, tau: float = 0.5
    ) -> pd.Series:
        return (int(y < fcst_c) - tau) * fcst
