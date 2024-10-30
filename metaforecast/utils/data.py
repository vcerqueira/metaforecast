import pandas as pd


class DataUtils:
    @staticmethod
    def train_test_split(df: pd.DataFrame, horizon: int):
        df_by_unq = df.groupby("unique_id")

        train_l, test_l = [], []
        for _, df_ in df_by_unq:
            df_ = df_.sort_values("ds")

            train_df_g = df_.head(-horizon)
            test_df_g = df_.tail(horizon)

            train_l.append(train_df_g)
            test_l.append(test_df_g)

        train_df = pd.concat(train_l).reset_index(drop=True)
        test_df = pd.concat(test_l).reset_index(drop=True)

        return train_df, test_df
