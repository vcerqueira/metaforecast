"""NeuralForecast extension for series-wise cross-validation.

Provides :class:`SeriesWiseNeuralForecast`, a subclass of
:class:`neuralforecast.NeuralForecast` whose ``cross_validation``
trains models on a designated subset of series (``train_uids``) and
optionally returns predictions only for ``test_uids``.
"""

from __future__ import annotations

import warnings
from typing import List

import numpy as np
import pandas as pd
import utilsforecast.processing as ufp
from coreforecast.grouped_array import GroupedArray
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import HuberIQLoss, IQLoss
from utilsforecast.compat import DataFrame, pl_DataFrame, pl_Series
from utilsforecast.validation import validate_freq


class SeriesWiseNeuralForecast(NeuralForecast):
    """NeuralForecast variant that fits on a subset of series.

    In standard ``NeuralForecast.cross_validation`` every model is fitted
    on the *entire* dataset (minus the hold-out window).  This subclass
    overrides the internal ``_no_refit_cross_validation`` so that models
    are fitted only on ``train_uids``.  If ``test_uids`` is given, the
    returned forecasts are restricted to those series.

    Parameters
    ----------
    train_uids : array-like
        Unique IDs used for training in this fold.
    test_uids : array-like, optional
        Unique IDs to keep in the CV output.  If ``None``, predictions
        for all series in ``df`` are returned.
    *args, **kwargs
        Forwarded to :class:`neuralforecast.NeuralForecast`.

    Examples
    --------
    >>> nf = SeriesWiseNeuralForecast(
    ...     models=[...], freq="ME", train_uids=train_uids, test_uids=test_uids
    ... )
    >>> cv = nf.cross_validation(df=df, val_size=h, test_size=h, n_windows=None)
    """

    def __init__(
        self,
        train_uids: np.ndarray,
        *args,
        test_uids: np.ndarray | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.train_uids = train_uids
        self.test_uids = test_uids

    def _no_refit_cross_validation(
        self,
        df: DataFrame | None,
        static_df: DataFrame | None,
        n_windows: int,
        step_size: int,
        val_size: int | None,
        test_size: int,
        verbose: bool,
        id_col: str,
        time_col: str,
        target_col: str,
        h: int,
        **data_kwargs,
    ) -> DataFrame:
        if df is None and not hasattr(self, "dataset"):
            raise ValueError("You must pass a DataFrame or have one stored.")

        if df is not None:
            validate_freq(df[time_col], self.freq)

            # Full dataset — used for prediction
            self.dataset, self.uids, self.last_dates, self.ds = self._prepare_fit(
                df=df,
                static_df=static_df,
                predict_only=False,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
            )

            # Training-only dataset — used for fitting
            train_df = df[df[id_col].isin(self.train_uids)].copy()
            self.train_dataset, *_ = self._prepare_fit(
                df=train_df,
                static_df=static_df,
                predict_only=False,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
            )
        else:
            id_col, time_col, target_col = (
                self.id_col,
                self.time_col,
                self.target_col,
            )
            if verbose:
                print("Using stored dataset.")

        if val_size is not None and self.dataset.min_size < (val_size + test_size):
            warnings.warn(
                "Validation and test sets are larger than the shortest series.", stacklevel=2
            )

        fcsts_df = ufp.cv_times(
            times=self.ds,
            uids=self.uids,
            indptr=self.dataset.indptr,
            h=h,
            test_size=test_size,
            step_size=step_size,
            id_col=id_col,
            time_col=time_col,
        )
        fcsts_df = ufp.sort(fcsts_df, [id_col, "cutoff", time_col])

        fcsts_list: List = []
        for model in self.models:
            if self._add_level and (
                model.loss.outputsize_multiplier > 1
                or isinstance(model.loss, (IQLoss, HuberIQLoss))
            ):
                continue

            # Fit on training subset, predict on the full dataset
            model.fit(
                dataset=self.train_dataset,
                val_size=val_size,
                test_size=test_size,
            )
            model_fcsts = model.predict(self.dataset, step_size=step_size, h=h, **data_kwargs)
            fcsts_list.append(model_fcsts)

        fcsts = np.concatenate(fcsts_list, axis=-1)

        effective_sizes = ufp.counts_by_id(fcsts_df, id_col)["counts"].to_numpy()
        needs_trim = effective_sizes.sum() != fcsts.shape[0]
        if self.scalers_ or needs_trim:
            indptr = np.arange(
                0,
                n_windows * h * (self.dataset.n_groups + 1),
                n_windows * h,
                dtype=np.int32,
            )
            if self.scalers_:
                fcsts = self._scalers_target_inverse_transform(fcsts, indptr)
            if needs_trim:
                trimmed = np.empty_like(fcsts, shape=(effective_sizes.sum(), fcsts.shape[1]))
                cv_indptr = np.append(0, effective_sizes).cumsum(dtype=np.int32)
                for i in range(fcsts.shape[1]):
                    ga = GroupedArray(fcsts[:, i], indptr)
                    trimmed[:, i] = ga._tails(cv_indptr)
                fcsts = trimmed

        self._fitted = True

        cols = self._get_model_names(add_level=self._add_level)
        if isinstance(self.uids, pl_Series):
            fcsts = pl_DataFrame(dict(zip(cols, fcsts.T)))
        else:
            fcsts = pd.DataFrame(fcsts, columns=cols)
        fcsts_df = ufp.horizontal_concat([fcsts_df, fcsts])

        result = ufp.join(
            fcsts_df,
            df[[id_col, time_col, target_col]],
            how="left",
            on=[id_col, time_col],
        )
        if self.test_uids is not None:
            result = ufp.filter_with_mask(result, ufp.is_in(result[id_col], self.test_uids))
        return result
