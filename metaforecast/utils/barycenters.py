import numpy as np
import pandas as pd

from tslearn.barycenters import (euclidean_barycenter,
                                 dtw_barycenter_averaging,
                                 dtw_barycenter_averaging_subgradient,
                                 softdtw_barycenter)


# def softdtw_barycenter2(X, **kwargs):
#     return softdtw_barycenter(X=X, init=euclidean_barycenter(X), **kwargs)
#
#
# def dtw_barycenter_averaging2(X, **kwargs):
#     return dtw_barycenter_averaging(X=X, init_barycenter=euclidean_barycenter(X), **kwargs)
#
#
# def dtw_barycenter_averaging_subgradient2(X, **kwargs):
#     return dtw_barycenter_averaging_subgradient(X=X, init_barycenter=euclidean_barycenter(X), **kwargs)


class BarycentricAveraging:
    """
    https://tslearn.readthedocs.io/en/latest/auto_examples/clustering/plot_barycenters.html#sphx-glr-auto-examples-clustering-plot-barycenters-py

    """
    BARYCENTERS = {
        'euclidean': euclidean_barycenter,
        'dtw': dtw_barycenter_averaging,
        'dtw_subgradient': dtw_barycenter_averaging_subgradient,
        # 'dtw_subgradient2': dtw_barycenter_averaging_subgradient2,
        'softdtw': softdtw_barycenter,
        # 'softdtw2': softdtw_barycenter2,
        # 'dtw2': dtw_barycenter_averaging2,
        # 'softdtw2': softdtw_barycenter,
    }

    BARYCENTER_PARAMS = {
        'euclidean': {},
        'dtw': {'max_iter': 5, 'tol': 1e-3},
        'dtw_subgradient': {'max_iter': 5, 'tol': 1e-3},
        'softdtw': {'gamma': .1, 'max_iter': 5, 'tol': 1e-3},
    }

    @classmethod
    def calc_average(cls, fcst: np.ndarray, barycenter: str):
        assert barycenter in [*cls.BARYCENTERS], 'Unknown barycenter'

        fcst_arr = cls.BARYCENTERS[barycenter](X=fcst,
                                               **cls.BARYCENTER_PARAMS[barycenter])

        fcst_arr = fcst_arr.flatten()

        return fcst_arr

    # @classmethod
    # def calc_average(cls, predictions: pd.DataFrame, barycenter: str):
    #     assert barycenter in [*cls.BARYCENTERS], 'Unknown barycenter'
    #
    #     preds_list = predictions.values.tolist()
    #
    #     preds_arr = cls.BARYCENTERS[barycenter](X=preds_list,
    #                                             **cls.BARYCENTER_PARAMS[barycenter])
    #
    #     preds = pd.Series(preds_arr.flatten(), index=predictions.columns)
    #
    #     return preds
