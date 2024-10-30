import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class Normalizations:
    @staticmethod
    def min_max_norm_vector(x: pd.Series) -> pd.Series:
        if not isinstance(x, pd.Series):
            x = pd.Series(x)

        scaler = MinMaxScaler()
        xn = scaler.fit_transform(x.values.reshape(-1, 1)).flatten()
        xn = pd.Series(xn, index=x.index)

        return xn

    @classmethod
    def normalize_and_proportion(cls, x):
        """Min max normalization followed by proportion"""
        nx = cls.min_max_norm_vector(x)
        out = nx / nx.sum()

        return out
