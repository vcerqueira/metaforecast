import typing

import numpy as np
import pandas as pd

from metaforecast.ensembles.base import Mixture

RowIdentifierType = typing.Union[int, typing.Hashable]


class MLewa(Mixture):
    """Dynamic ensemble using exponentially weighted averaging (EWA).


    Implementation inspired by R's opera package, this class combines forecasts using
    online learning with exponential weights. Weights are updated based on recent
    performance to adapt to changing patterns.

    See Also
    --------
    Mixture : Parent class implementing core ensemble functionality
    opera : R package with original implementation

    Notes
    -----
    This implementation follows the EWA algorithm described in [1] and [2]

    References
    -----------
    [1] Cesa-Bianchi, N., & Lugosi, G. (2006). "Prediction, learning, and games."
    Cambridge University Press.

    [2] Gaillard, P., & Goude, Y. (2015).
    "Forecasting electricity consumption by aggregating experts."
    In Modeling and Stochastic Learning for Forecasting in High Dimensions
    (pp. 95-115). Springer, Cham.

    [3] Cerqueira, V., Torgo, L., Pinto, F., & Soares, C. (2019).
    "Arbitrage of forecasting experts." Machine Learning, 108, 913-944.

    Examples
    ---------
    >>> from datasetsforecast.m3 import M3
    >>> from neuralforecast import NeuralForecast
    >>> from neuralforecast.models import NHITS, NBEATS, MLP
    >>> from metaforecast.ensembles import MLewa
    >>>
    >>> df, *_ = M3.load('.', group='Monthly')
    >>>
    >>> CONFIG = {'input_size': 12,
    >>>           'h': 12,
    >>>           'accelerator': 'cpu',
    >>>           'max_steps': 10, }
    >>>
    >>> models = [
    >>>     NBEATS(**CONFIG, stack_types=3 * ["identity"]),
    >>>     NHITS(**CONFIG),
    >>>     MLP(**CONFIG),
    >>>     MLP(num_layers=3, **CONFIG),
    >>> ]
    >>>
    >>> nf = NeuralForecast(models=models, freq='M')
    >>>
    >>> # cv to build meta-data
    >>> n_windows = df['unique_id'].value_counts().min()
    >>> n_windows = int(n_windows // 2)
    >>> fcst_cv = nf.cross_validation(df=df, n_windows=n_windows, step_size=1)
    >>> fcst_cv = fcst_cv.reset_index()
    >>> fcst_cv = fcst_cv.groupby(['unique_id', 'cutoff']).head(1).drop(columns='cutoff')
    >>>
    >>> # fitting combination rule
    >>> ensemble = MLewa(loss_type='square', gradient=True, trim_ratio=.8)
    >>> ensemble.fit(fcst_cv)
    >>>
    >>> # re-fitting models
    >>> nf.fit(df=df)
    >>>
    >>> # forecasting and combining
    >>> fcst = nf.predict()
    >>> fcst_ensemble = ensemble.predict(fcst.reset_index())

    """

    def __init__(
        self,
        loss_type: str,
        gradient: bool,
        weight_by_uid: bool = False,
        trim_ratio: float = 1,
    ):
        """Initialize online ensemble with exponential weighting strategy.


        Parameters
        ----------
        loss_type : {'square', 'pinball', 'percentage', 'absolute', 'log'}
            Loss function for evaluating and weighting ensemble members:
            - square: Mean squared error
            - pinball: Quantile loss
            - percentage: Mean absolute percentage error
            - absolute: Mean absolute error
            - log: Log loss

        gradient : bool, default=False
            If True, use gradient for weight updates

        weight_by_uid : bool, default=True
            Whether to compute weights separately for each series:
            - True: Individual weights per series (may be computationally intensive)
            - False: Global weights across all series

        trim_ratio : float, default=1.0
            Proportion of models to retain in ensemble, between 0 and 1:
            - 1.0: Keep all models
            - 0.5: Keep top 50% of models
            Models are selected based on validation performance

        """

        super().__init__(
            loss_type=loss_type,
            gradient=gradient,
            trim_ratio=trim_ratio,
            weight_by_uid=weight_by_uid,
        )

        self.alias = "MLewa"

    def _update_mixture(self, fcst: pd.DataFrame, y: np.ndarray, **kwargs):
        """_update_mixture

        Updating the weights of the ensemble

        :param fcst: predictions of the ensemble members (columns) in different time steps (rows)
        :type fcst: pd.DataFrame

        :param y: actual values of the time series
        :type y: np.ndarray

        :return: self
        """

        for i, fc in fcst.iterrows():
            w = self._weights_from_regret(iteration=i)

            self.weights[i], self.ensemble_fcst[i] = self._calc_ensemble_fcst(fc, w)

            loss_experts = self._calc_loss(
                fcst=fc, y=y[i], fcst_c=self.ensemble_fcst[i]
            )
            loss_mixture = self._calc_loss(
                fcst=self.ensemble_fcst[i],
                y=y[i],
                fcst_c=self.ensemble_fcst[i],
            )

            regret_i = loss_mixture - loss_experts

            # update regret
            for mod in self.regret:
                self.regret[mod] += regret_i[mod]

            n = len(self.model_names)

            eta_update = np.sqrt(
                np.log(n) / (np.log(n) / self.eta[i] ** 2 + regret_i**2)
            )
            self.eta[int(str(i)) + 1] = eta_update

    def _weights_from_regret(self, iteration: RowIdentifierType = -1, **kwargs):
        """_weights_from_regret

        Updating the weights based on regret minimization

        """
        curr_regret = np.array(list(self.regret.values()))

        if np.max(curr_regret) > 0:
            w = self._truncate_loss(np.exp(self.eta[iteration] * curr_regret))
            w /= np.sum(w)
        else:
            w = np.ones_like(curr_regret) / len(curr_regret)

        w = pd.Series(w, index=self.model_names)

        return w

    def _initialize_params(self, fcst: pd.DataFrame):
        n_row, n_col = fcst.shape[0], len(self.model_names)

        self.eta = np.full(shape=(n_row + 1, n_col), fill_value=np.exp(350))
        self.regret = {k: 0 for k in self.model_names}
        self.weights = np.zeros((n_row, n_col))
        self.ensemble_fcst = np.zeros(n_row)

    def update_weights(self, fcst: pd.DataFrame):
        raise NotImplementedError

    @staticmethod
    def _truncate_loss(x):
        return np.clip(x, np.exp(-700), np.exp(700))
