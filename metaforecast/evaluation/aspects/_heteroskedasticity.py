import pandas as pd
import statsmodels.stats.api as sms
from statsmodels.formula.api import ols


class Heteroskedasticity:
    """
    Class for performing heteroskedasticity tests on time series data.

    Methods:
        het_tests(series: pd.Series, test: str) -> float:
            Tests for heteroskedasticity using specified test.
        get_ols_residuals(series: pd.Series) -> pd.Series:
            Gets the residuals from OLS regression.
        get_ols_model(series: pd.Series) -> ols:
            Fits an OLS model to the series.
        run_all_tests(series: pd.Series) -> dict:
            Runs all heteroskedasticity tests and returns a dictionary of p-values.
    """

    TESTS = {
        "white": "White",
        "breuschpagan": "Breusch-Pagan",
        "breakvar": "Goldfeld-Quandt",
    }

    TEST_NAMES = [*TESTS.values()]

    @classmethod
    def het_tests(cls, series: pd.Series, test: str):
        """
        Tests for heteroskedasticity using specified test.

        Args:
            series (pd.Series): Univariate time series.
            test (str): String denoting the test. One of 'white', 'goldfeldquandt', or 'breuschpagan'.

        Returns:
            float: p-value of the test.
        """
        assert test in cls.TEST_NAMES, "Unknown test"

        mod = cls.get_ols_model(series)

        if test == "White":
            _, p_value, _, _ = sms.het_white(mod.resid, mod.model.exog)
        elif test == "Goldfeld-Quandt":
            _, p_value, _ = sms.het_goldfeldquandt(
                mod.resid, mod.model.exog, alternative="two-sided"
            )
        else:
            _, p_value, _, _ = sms.het_breuschpagan(mod.resid, mod.model.exog)

        return p_value

    @classmethod
    def get_ols_residuals(cls, series: pd.Series):
        """
        Gets the residuals from OLS regression.

        Args:
            series (pd.Series): Univariate time series.

        Returns:
            pd.Series: Residuals from OLS regression.
        """

        return cls.get_ols_model(series).resid

    @classmethod
    def get_ols_model(cls, series: pd.Series):
        """
        Fits an OLS model to the series.

        Args:
            series (pd.Series): Univariate time series.

        Returns:
            ols: Fitted OLS model.
        """

        series = series.reset_index(drop=True).reset_index()
        series.columns = ["time", "value"]
        series["time"] += 1

        olsr = ols("value ~ time", series).fit()

        return olsr

    @classmethod
    def run_all_tests(cls, series: pd.Series):
        """
        Runs all heteroskedasticity tests and returns a dictionary of p-values.

        Args:
            series (pd.Series): Univariate time series.

        Returns:
            dict: Dictionary containing p-values of all tests.
        """

        test_results = {k: cls.het_tests(series, k) for k in cls.TEST_NAMES}

        return test_results
