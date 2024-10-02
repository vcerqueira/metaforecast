import pandas as pd

from metaforecast.ensembles.windowing import Windowing


class BestOnTrain(Windowing):

    def __init__(self, select_by_uid: bool):
        self.alias = 'BestOnTrain'

        super().__init__(freq='',
                         select_best=True,
                         trim_ratio=1,
                         weight_by_uid=select_by_uid)

        self.use_window = False
        self.select_by_uid = select_by_uid

    def update_weights(self, **kwargs):
        raise NotImplementedError


class LossOnTrain(Windowing):

    def __init__(self, trim_ratio: float, weight_by_uid: bool):
        self.alias = 'LossOnTrain'

        super().__init__(freq='',
                         select_best=False,
                         trim_ratio=trim_ratio,
                         weight_by_uid=weight_by_uid)

        self.use_window = False

    def update_weights(self, **kwargs):
        raise NotImplementedError


class EqAverage(Windowing):

    def __init__(self, trim_ratio: float, select_by_uid: bool):
        self.alias = 'EqAverage'

        super().__init__(freq='',
                         select_best=False,
                         trim_ratio=trim_ratio,
                         weight_by_uid=select_by_uid)

        self.use_window = False

    def update_weights(self, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _weights_from_errors(scores: pd.Series) -> pd.Series:
        weights = pd.Series({k: 1 / len(scores) for k in scores.index})

        return weights
