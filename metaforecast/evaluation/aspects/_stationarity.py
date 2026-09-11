import warnings

import pandas as pd
from arch.unitroot import PhillipsPerron
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.api import STL
from statsmodels.tsa.stattools import adfuller, kpss

warnings.simplefilter("ignore", InterpolationWarning)


class DifferencingTests:
    """
    Seasonal and non-seasonal differencing tests.

    can include more nuance based on tests
    # https://medium.com/towards-data-science/understanding-time-series-trend-addfd9d7764e


    Methods:
        nsdiffs(series: pd.Series, period: int, test: str = 'seas') -> int:
            Estimate number of seasonal differences required for seasonal stationarity.
        ndiffs(series: pd.Series, test: str = 'kpss', test_type: str = 'trend') -> int:
            Estimate number of differences required for non-seasonal stationarity.
        _check_stationarity(series: pd.Series, test: str, test_type: str) -> bool:
            Check if series is stationary using specified test.
        _wang_smith_hyndman_test(series: pd.Series, period: int) -> int:
            Implementation of Wang-Smith-Hyndman seasonal strength test.
        _ocsb_test(series: pd.Series, period: int) -> int:
            Simplified OCSB test.
    """

    NSDIFF_TESTS = {
        "seas": "Wang-Smith-Hyndman",
        "ocsb": "OCSB",
        # 'ch': 'Canova-Hansen'
    }

    NDIFF_TESTS = {"kpss": "KPSS", "adf": "Augmented Dickey-Fuller", "pp": "Philips-Perron"}

    TEST_TYPES = ["trend", "level"]

    @staticmethod
    def nsdiffs(series: pd.Series, period: int, test: str = "seas") -> int:
        """
        Estimate number of seasonal differences required for seasonal stationarity.

        Parameters:
        -----------
        series : pd.Series
            Time series data.
        period : int
            Seasonal period.
        test : str
            Type of test to use ('seas' or 'ocsb').

        Returns:
        --------
        int : Recommended number of seasonal differences.
        """

        if test not in DifferencingTests.NSDIFF_TESTS:
            raise ValueError(
                f"Unknown test type. Must be one of {[*DifferencingTests.NSDIFF_TESTS]}"
            )

        if test == "seas":
            n_diffs = DifferencingTests._wang_smith_hyndman_test(series, period)
        else:
            n_diffs = DifferencingTests._ocsb_test(series, period)

        return n_diffs

    @staticmethod
    def ndiffs(series: pd.Series, test: str = "kpss", test_type: str = "trend") -> int:
        """
        Estimate number of differences required for non-seasonal stationarity.

        Parameters:
        -----------
        series : pd.Series
            Time series data.
        test : str
            Type of test to use ('kpss', 'adf', or 'pp').
        test_type : str
            Type of test ('trend' or 'level').

        Returns:
        --------
        int : Recommended number of differences.
        """
        if test not in DifferencingTests.NDIFF_TESTS:
            raise ValueError(
                f"Unknown test type. Must be one of {[*DifferencingTests.NDIFF_TESTS]}"
            )

        if test_type not in DifferencingTests.TEST_TYPES:
            raise ValueError(f"Unknown test_type. Must be one of {DifferencingTests.TEST_TYPES}")

        max_d = min(2, int(len(series) / 3))  # Maximum number of differences
        d = 0
        while d <= max_d:
            is_stationary = DifferencingTests._check_stationarity(series, test, test_type)
            if is_stationary:
                return d

            series = series.diff().dropna()
            d += 1

        return d

    @staticmethod
    def _check_stationarity(series: pd.Series, test: str, test_type: str) -> bool:
        """Check if series is stationary using specified test"""

        if test == "kpss":
            regression = "ct" if test_type == "trend" else "c"
            _, p_value, *_ = kpss(series, regression=regression)
            return p_value > 0.05

        if test == "adf":
            regression = "ct" if test_type == "trend" else "c"
            _, p_value, *_ = adfuller(series, regression=regression)
            return p_value < 0.05

        regression = "ct" if test_type == "trend" else "c"
        test = PhillipsPerron(y=series, trend=regression)
        return test.pvalue < 0.05

    @staticmethod
    def _wang_smith_hyndman_test(series: pd.Series, period: int) -> int:
        """Implementation of Wang-Smith-Hyndman seasonal strength test"""

        series_decomp = STL(series, period=period).fit()

        # variance of residuals + seasonality
        resid_seas_var = (series_decomp.resid + series_decomp.seasonal).var()
        # variance of residuals
        resid_var = series_decomp.resid.var()

        # Calculate seasonal strength
        seasonal_strength = 1 - (resid_var / resid_seas_var)

        # If seasonal strength is greater than 0.64, suggest seasonal differencing
        ndiffs = 1 if seasonal_strength > 0.64 else 0

        return ndiffs

    @staticmethod
    def _ocsb_test(series: pd.Series, period: int) -> int:
        """
        Simplified OCSB test
        """
        # seasonal differences
        seasonal_diff = series.diff(period).dropna()

        # Perform ADF test on seasonal differences
        _, p_value, *_ = adfuller(seasonal_diff)

        # If p-value < 0.05, series needs seasonal differencing
        ndiffs = 1 if p_value < 0.05 else 0

        return ndiffs
