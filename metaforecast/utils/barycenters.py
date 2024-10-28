import numpy as np
from tslearn.barycenters import (
    dtw_barycenter_averaging,
    dtw_barycenter_averaging_subgradient,
    euclidean_barycenter,
    softdtw_barycenter,
)


class BarycentricAveraging:
    """
    BarycentricAveraging

    Averaging a set of time series (or time series subsequences) using barycentric averaging
    Wrapper for tslearn's functions

    >>> import numpy as np
    >>> from metaforecast.utils.barycenters import BarycentricAveraging
    >>> x = np.random.random((20, 5))
    >>>
    >>> x_avg = BarycentricAveraging.calc_average(x, 'dtw')
    >>> # print(x_avg)

    """

    BARYCENTERS = {
        "euclidean": euclidean_barycenter,
        "dtw": dtw_barycenter_averaging,
        "dtw_subgradient": dtw_barycenter_averaging_subgradient,
        "softdtw": softdtw_barycenter,
    }

    BARYCENTER_PARAMS = {
        "euclidean": {},
        "dtw": {"max_iter": 5, "tol": 1e-3},
        "dtw_subgradient": {"max_iter": 5, "tol": 1e-3},
        "softdtw": {"gamma": 0.1, "max_iter": 5, "tol": 1e-3},
    }

    @classmethod
    def calc_average(cls, fcst: np.ndarray, barycenter: str):
        """calc_average

        Compute the barycentric averaging of a set of time series subsequences (or forecasts)

        :param fcst: time series subsequences or forecasts as an array-like structure
        :type fcst: np.ndarray

        :param barycenter: Barycenter used for averaging. One of "euclidean", "dtw",
        "dtw_subgradient", or "softdtw"
        :type barycenter: str

        :return: averaged subsequences as np.ndarray
        """
        assert barycenter in [*cls.BARYCENTERS], "Unknown barycenter"

        fcst_arr = cls.BARYCENTERS[barycenter](
            X=fcst, **cls.BARYCENTER_PARAMS[barycenter]
        )

        fcst_arr = fcst_arr.flatten()

        return fcst_arr
