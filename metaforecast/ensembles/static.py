import pandas as pd

from metaforecast.ensembles.windowing import Windowing


class BestOnTrain(Windowing):
    """ BestOnTrain

    Selecting the ensemble member with the best performance on training data

    """

    def __init__(self, select_by_uid: bool):
        """

        :param select_by_uid: whether to select the best ensemble member by unique_id (True) or across all dataset (False)
        :type select_by_uid: bool
        """

        super().__init__(freq='',
                         select_best=True,
                         trim_ratio=1,
                         weight_by_uid=select_by_uid)

        self.alias = 'BestOnTrain'

        self.use_window = False
        self.select_by_uid = select_by_uid

    def update_weights(self, **kwargs):
        raise NotImplementedError


class LossOnTrain(Windowing):
    """ LossOnTrain

    Weighting the ensemble members according to the squared error on training data

    """

    def __init__(self, trim_ratio: float, weight_by_uid: bool):
        """

        :param weight_by_uid: Whether to weight the ensemble by unique_id (True) or dataset (False)
        Defaults to True, but this can become computationally demanding for datasets with a large number of time series
        :type weight_by_uid: bool

        :param trim_ratio: Ratio (0-1) of ensemble members to keep in the ensemble.
        (1-trim_ratio) of models will not be used during inference based on validation accuracy.
        Defaults to 1, which means all ensemble members are used.
        :type trim_ratio: float

        """
        super().__init__(freq='',
                         select_best=False,
                         trim_ratio=trim_ratio,
                         weight_by_uid=weight_by_uid)

        self.alias = 'LossOnTrain'

        self.use_window = False

    def update_weights(self, **kwargs):
        raise NotImplementedError


class EqAverage(Windowing):
    """ EqAverage

    Combining ensemble members with a simple average after a preliminary trimming

    """

    def __init__(self, trim_ratio: float = 1, select_by_uid: bool = True):
        """
        :param select_by_uid: Whether to trim the ensemble by unique_id (True) or dataset (False)
        :type select_by_uid: bool

        :param trim_ratio: Ratio (0-1) of ensemble members to keep in the ensemble.
        (1-trim_ratio) of models will not be used during inference based on validation accuracy.
        Defaults to 1, which means all ensemble members are used.
        :type trim_ratio: float

        """
        super().__init__(freq='',
                         select_best=False,
                         trim_ratio=trim_ratio,
                         weight_by_uid=select_by_uid)

        self.alias = 'EqAverage'

        self.use_window = False

    def update_weights(self, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _weights_from_errors(scores: pd.Series) -> pd.Series:
        weights = pd.Series({k: 1 / len(scores) for k in scores.index})

        return weights
