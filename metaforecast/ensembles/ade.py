from typing import Union, Tuple, List, Optional

import pandas as pd
import lightgbm as lgb
from mlforecast import MLForecast
from statsforecast import StatsForecast

from sklearn.multioutput import MultiOutputRegressor as MIMO

from metaforecast.utils.normalization import Normalizations
from metaforecast.ensembles.base import BaseADE

DForDFTuple = Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]


class ADE(BaseADE):
    LGB_PARS = {'verbosity': -1, 'n_jobs': 1, 'linear_tree': True}
    MLF_PREPROCESS_PARS = {'static_features': []}

    def __init__(self,
                 freq: str,
                 trim_ratio: float,
                 meta_lags: List[int],
                 trim_by_uid: bool = True,
                 meta_model=MIMO(lgb.LGBMRegressor(**LGB_PARS))):
        """
        :param trim_ratio:
        :param meta_model:
        """
        self.frequency = freq

        super().__init__(window_size=self.WINDOW_SIZE_BY_FREQ[self.frequency],
                         trim_ratio=trim_ratio,
                         trim_by_uid=trim_by_uid,
                         meta_model=meta_model)

        self.model_names = None

        self.meta_lags = meta_lags
        self.lag_names = [f'lag{i}' for i in self.meta_lags]

        self.meta_mlf = MLForecast(
            models=[],
            freq=self.frequency,
            lags=self.meta_lags
        )

        self.meta_df = None
        self.raw_meta_data = None
        self.insample_scores = None
        self.use_window = False

    def fit(self, insample_fcst: pd.DataFrame, **kwargs):
        """

        """

        self._fit(insample_fcst)

    def _fit(self, insample_fcst):
        if self.model_names is None:
            self.model_names = insample_fcst.columns.to_list()
            self.model_names = [x for x in self.model_names if x not in self.METADATA + ['h']]

        self._set_n_models()

        in_sample_loss_df = self._get_insample_loss(insample_fcst)

        self.insample_scores = self.evaluate_base_fcst(insample_fcst=insample_fcst,
                                                       use_window=self.use_window)

        self.raw_meta_data = self.meta_mlf.preprocess(in_sample_loss_df, **self.MLF_PREPROCESS_PARS)

        self.meta_df = self._process_meta_data(self.raw_meta_data)

        x, y = self.meta_df
        # print(y.isna().mean())
        if y.isna().any().any():
            y = y.ffill().bfill()

        self.meta_model.fit(x, y)

    def predict(self, preds: pd.DataFrame, train: pd.DataFrame, h: int):

        fcst = self._predict(preds=preds, train=train, h=h)
        fcst.name = self.alias

        return fcst

    def update_weights(self, fcst: pd.DataFrame):
        raise NotImplementedError

    def _predict(self, preds: pd.DataFrame, train: pd.DataFrame, h: int):
        # could use ade.mlf.make_future_dataframe(h=4)
        df_ext = train.merge(preds, on=['unique_id', 'ds'], how='outer')
        df_ext = df_ext[self.METADATA]
        df_ext['y'] = df_ext['y'].fillna(value=-1)

        meta_dataset = self.meta_mlf.preprocess(df_ext, **self.MLF_PREPROCESS_PARS)

        weights = self._weights_by_uid(meta_dataset, h=h)

        fcst = preds.apply(lambda x: self._weighted_average(x, weights), axis=1)

        return fcst

    def _get_insample_loss(self, insample_fcst: pd.DataFrame):
        in_sample_loss = []
        in_sample_uid = insample_fcst.copy().groupby('unique_id')
        for uid, uid_df in in_sample_uid:
            for mod in self.model_names:
                uid_df[mod] = uid_df[mod] - uid_df['y']

            in_sample_loss.append(uid_df)

        in_sample_loss_df = pd.concat(in_sample_loss)

        # first h forward
        # could average all horizons
        if 'h' in in_sample_loss_df.columns:
            in_sample_loss_df = in_sample_loss_df.query('h==1').drop(columns=['h'])

        return in_sample_loss_df

    def _process_meta_data(self,
                           meta_data: pd.DataFrame,
                           return_X_y: bool = True) -> DForDFTuple:

        lag_locs = meta_data.columns.str.startswith('lag')
        lag_cols = meta_data.columns[lag_locs].to_list()

        if return_X_y:
            X_meta, Y_meta = meta_data[lag_cols], meta_data[self.model_names]
            return X_meta, Y_meta
        else:
            meta_df = meta_data[lag_cols + self.model_names]

        return meta_df

    def _weights_by_uid(self, df: pd.DataFrame, h: int):
        top_overall = self._get_top_k(self.insample_scores.mean())
        top_by_uid = self.insample_scores.apply(self._get_top_k, axis=1)

        uid_weights = {}
        for uid, meta_uid_df in df.groupby('unique_id'):
            if h > 1:
                lags = meta_uid_df.head(-(h - 1)).tail(1)[self.lag_names]
            else:
                lags = meta_uid_df.tail(1)[self.lag_names]

            meta_pred = self.meta_model.predict(lags)
            meta_pred = pd.DataFrame(meta_pred, columns=self.model_names)

            weights = self._weights_from_errors(meta_pred)

            if self.trim_by_uid:
                poor_models = [x not in top_by_uid[uid] for x in weights.index]
            else:
                poor_models = [x not in top_overall for x in weights.index]

            weights[poor_models] = 0
            weights /= weights.sum()

            uid_weights[uid] = weights

        weights_df = pd.DataFrame(uid_weights).T
        weights_df.index.name = 'unique_id'

        return weights_df

    def _reweight_by_redundancy(self):
        raise NotImplementedError

    @staticmethod
    def _weights_from_errors(meta_predictions: pd.DataFrame) -> pd.Series:
        e_hat = meta_predictions.abs()

        W = e_hat.apply(
            func=lambda x: Normalizations.normalize_and_proportion(-x),
            axis=1)

        weight_s = W.iloc[0]

        return weight_s


class GlobalADE(ADE):
    ## todo in inference lags are the same of all
    # could include past errors...

    def __init__(self,
                 freq: str,
                 trim_ratio: float,
                 meta_lags: List[int],
                 trim_by_uid: bool = True,
                 meta_model=lgb.LGBMRegressor(**ADE.LGB_PARS)):

        super().__init__(freq=freq,
                         trim_ratio=trim_ratio,
                         meta_lags=meta_lags,
                         trim_by_uid=trim_by_uid,
                         meta_model=meta_model)

        self.alias = 'GADE'

    def _process_meta_data(self,
                           meta_data: pd.DataFrame,
                           return_X_y: bool = True) -> DForDFTuple:

        lag_locs = meta_data.columns.str.startswith('lag')
        lag_cols = meta_data.columns[lag_locs].to_list()

        df_melt = meta_data.drop(columns='y').melt(['unique_id', 'ds'] + lag_cols)
        df_melt['unique_id'] = df_melt.apply(lambda x: f'{x["unique_id"]}_{x["variable"]}', axis=1)
        df_melt = df_melt.rename(columns={'value': 'error'})

        if return_X_y:
            X_meta, Y_meta = df_melt[lag_cols], df_melt['error']
            return X_meta, Y_meta
        else:
            meta_df = df_melt[lag_cols + ['error']]

        return meta_df

    def _weights_by_uid(self, df: pd.DataFrame, h: int):
        top_overall = self._get_top_k(self.insample_scores.mean())
        top_by_uid = self.insample_scores.apply(self._get_top_k, axis=1)

        uid_weights = {}
        for uid, meta_uid_df in df.groupby('unique_id'):
            lags = meta_uid_df.head(-(h - 1)).tail(1)[self.lag_names]

            meta_pred = self.meta_model.predict(lags)
            meta_pred = pd.DataFrame(meta_pred, columns=self.model_names)

            weights = self._weights_from_errors(meta_pred)

            if self.trim_by_uid:
                poor_models = [x not in top_by_uid[uid] for x in weights.index]
            else:
                poor_models = [x not in top_overall for x in weights.index]

            weights[poor_models] = 0
            weights /= weights.sum()

            uid_weights[uid] = weights

        weights_df = pd.DataFrame(uid_weights).T
        weights_df.index.name = 'unique_id'

        return weights_df

    def _reweight_by_redundancy(self):
        raise NotImplementedError

    def update_weights(self, fcst: pd.DataFrame):
        raise NotImplementedError


class MLForecastADE(ADE):

    def __init__(self,
                 mlf: MLForecast,
                 trim_ratio: float,
                 sf: Optional[StatsForecast] = None,
                 meta_model=MIMO(lgb.LGBMRegressor(**ADE.LGB_PARS))):
        """
        :param trim_ratio:
        :param meta_model:
        """
        self.mlf = mlf
        self.sf = sf
        self.frequency = self.mlf.ts.freq

        super().__init__(freq=self.frequency,
                         trim_ratio=trim_ratio,
                         meta_model=meta_model,
                         meta_lags=self.mlf.ts.lags)

    def fit(self, **kwargs):
        """

        """

        insample_fcst = self.mlf.fcst_fitted_values_

        if self.sf is not None:
            self.sf.forecast(fitted=True, h=1)
            insample_fcst_sf = self.sf.forecast_fitted_values()

            insample_fcst = insample_fcst.merge(insample_fcst_sf.drop(columns='y'),
                                                on=self.METADATA_NO_T)

        self._fit(insample_fcst)

    def predict(self, train: pd.DataFrame, h: int, **kwargs):
        base_fcst = self.mlf.predict(h=h)

        if self.sf is not None:
            base_fcst_sf = self.sf.predict(h=h)

            base_fcst = base_fcst.merge(base_fcst_sf, on=self.METADATA_NO_T)

        fcst = self._predict(preds=base_fcst, train=train, h=h)

        return fcst

    def update_weights(self, fcst: pd.DataFrame):
        raise NotImplementedError

    def _reweight_by_redundancy(self):
        raise NotImplementedError

    def update_estimates(self, df: pd.DataFrame):
        """
        Updating loss statistics for dynamic model selection

        :param df: dataset with actual values and predictions, similar to insample predictions
        """

        raise NotImplementedError
