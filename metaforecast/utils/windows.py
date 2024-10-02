import re

import pandas as pd
import numpy as np


def create_windows(input_size, horizon, series: pd.Series) -> pd.DataFrame:
    if series.name is None:
        name = 'Series'
    else:
        name = series.name

    input_shifts = list(range(input_size - 1, -1, -1))
    output_shifts = list(range(-1, -(horizon + 1), -1))
    shifts_levels = input_shifts + output_shifts

    shifted_series = [series.shift(i) for i in shifts_levels]

    rec_df = pd.concat(shifted_series, axis=1).dropna()
    column_names = []
    for i in shifts_levels:
        if i >= 0:
            col = f'{name}(L-{i})'
        else:
            col = f'{name}(T-{np.abs(i)})'

        column_names.append(col)

    rec_df.columns = column_names
    # rec_df.index.name = f'{rec_df.index.name} (at L-0)'

    return rec_df


def time_delay_embedding(series: pd.Series,
                         n_lags: int,
                         horizon: int,
                         return_Xy: bool = False):
    """
    Time delay embedding

    Time series for supervised learning

    :param series: time series as pd.Series
    :param n_lags: number of past values to used as explanatory variables
    :param horizon: how many values to forecast
    :param return_Xy: whether to return the lags split from future observations

    :return: pd.DataFrame with reconstructed time series
    """
    assert isinstance(series, pd.Series)

    if series.name is None:
        name = 'Series'
    else:
        name = series.name

    n_lags_iter = list(range(n_lags, -horizon, -1))

    df_list = [series.shift(i) for i in n_lags_iter]
    df = pd.concat(df_list, axis=1).dropna()
    df.columns = [f'{name}(t-{j - 1})'
                  if j > 0 else f'{name}(t+{np.abs(j) + 1})'
                  for j in n_lags_iter]

    df.columns = [re.sub('t-0', 't', x) for x in df.columns]

    if not return_Xy:
        return df

    is_future = df.columns.str.contains('\+')

    X = df.iloc[:, ~is_future]
    Y = df.iloc[:, is_future]
    if Y.shape[1] == 1:
        Y = Y.iloc[:, 0]

    return X, Y
