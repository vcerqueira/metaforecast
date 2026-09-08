"""Homomorphic-Controlled Augmentation (HCA) for time series.

Leverages homomorphic decomposition to disentangle a time series into three
interpretable components -- the phase spectrum, the spectral envelope, and
harmonic details -- and recombines them with a randomly selected *style*
series from the training set.  The phase spectrum is preserved to maintain
temporal structure, while the envelope and harmonics are independently
perturbed to simulate realistic distributional drift.

Based on Li et al. (2026) [1]_.

References
----------
.. [1] Li, H., Cheng, L., Liu, X., Liu, Z., Long, L., Zhang, Y., & Dai, F.
   (2026). "Homomorphic-Controlled Augmentation for Time Series
   Forecasting."  *ICASSP 2026*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from metaforecast.synth.generators.base import SemiSyntheticTransformer


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Causal moving-average low-pass filter on a 1-D signal."""
    if window <= 1:
        return x.copy()
    kernel = np.ones(window) / window
    padded = np.pad(x, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(x)]


class HomomorphicAugmentation(SemiSyntheticTransformer):
    """Augment time series via homomorphic decomposition and controlled recombination.

    For each *content* series in the dataset, a *style* series is randomly
    chosen from the same dataset.  Both are decomposed into:

    * **Phase spectrum** -- preserved from the content series to maintain
      temporal structure and causal flow.
    * **Spectral envelope** *E(w)* -- the low-frequency shape of the
      log-magnitude spectrum (global statistics).
    * **Harmonic details** *D(w)* -- the residual capturing local periodic
      structures.

    A new magnitude spectrum is synthesised by interpolating the harmonic
    details and shifting the envelope, then reconstructed with the content
    phase to produce the augmented series.

    Parameters
    ----------
    delta : float, default 0.3
        Harmonic mixing ratio in [0, 1].  Controls the balance between
        content and style harmonics: 0 keeps content harmonics; 1 fully
        replaces with style.
    lambda_ : float, default 1.0
        Drift strength (>= 0).  Controls the influence of the normalised
        style envelope on the content envelope.  Larger values induce more
        pronounced global modifications.
    lpf_window : int, default 5
        Window size for the moving-average low-pass filter applied to the
        log-magnitude spectrum to extract the spectral envelope.
    rename_uids : bool, default True
        Whether to create new identifiers for augmented series.

    Examples
    --------
    >>> from metaforecast.synth import HomomorphicAugmentation
    >>> aug = HomomorphicAugmentation(delta=0.3, lambda_=1.0)
    >>> augmented_df = aug.transform(train_df)
    """

    def __init__(
        self,
        delta: float = 0.3,
        lambda_: float = 1.0,
        lpf_window: int = 5,
        rename_uids: bool = True,
    ):
        super().__init__(alias="HCA", rename_uids=rename_uids)

        self.delta = delta
        self.lambda_ = lambda_
        self.lpf_window = lpf_window

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Augment each series using a randomly chosen style series.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset with columns ``unique_id``, ``ds``, ``y``.

        Returns
        -------
        pd.DataFrame
            Augmented dataset with the same structure.
        """
        self._assert_datatypes(df)

        groups = {uid: uid_df for uid, uid_df in df.groupby(self.id_col)}
        uids = list(groups.keys())

        df_t_list = []
        for uid in uids:
            content_df = groups[uid]

            style_uid = uid
            while style_uid == uid and len(uids) > 1:
                style_uid = uids[np.random.randint(0, len(uids))]
            style_df = groups[style_uid]

            ts_df = self._create_synthetic_ts(content_df, style_df=style_df)

            if self.rename_uids:
                ts_df[self.id_col] = ts_df[self.id_col].apply(
                    lambda x: f"{x}_{self.alias}{self.counter}"
                )
            self.counter += 1
            df_t_list.append(ts_df)

        return pd.concat(df_t_list).reset_index(drop=True)

    def _create_synthetic_ts(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df_ = df.copy()
        y_content = df_[self.target_col].values.astype(np.float64)

        style_df = kwargs.get("style_df")
        if style_df is not None:
            y_style = style_df[self.target_col].values.astype(np.float64)
        else:
            y_style = y_content

        T = len(y_content)
        style_len = len(y_style)
        if style_len != T:
            if style_len > T:
                y_style = y_style[:T]
            else:
                y_style = np.pad(y_style, (0, T - style_len), mode="wrap")

        augmented = self._hca_core(y_content, y_style)

        df_.loc[:, self.target_col] = augmented.astype(df_[self.target_col].dtype)
        return df_

    def _hca_core(self, x_content: np.ndarray, x_style: np.ndarray) -> np.ndarray:
        """Core HCA: decompose, recombine, reconstruct."""
        T = len(x_content)
        eps = 1e-10

        Fc = np.fft.rfft(x_content)
        Fs = np.fft.rfft(x_style)

        Ac = np.abs(Fc) + eps
        As = np.abs(Fs) + eps
        phi_c = np.angle(Fc)

        log_Ac = np.log(Ac)
        log_As = np.log(As)

        Ec = _moving_average(log_Ac, self.lpf_window)
        Dc = log_Ac - Ec

        Es = _moving_average(log_As, self.lpf_window)
        Ds = log_As - Es

        D_hat = (1.0 - self.delta) * Dc + self.delta * Ds

        mu_s = Es.mean()
        sigma_s = Es.std() + eps
        Es_norm = (Es - mu_s) / sigma_s

        log_A_aug = Ec + self.lambda_ * Es_norm + D_hat
        A_aug = np.exp(log_A_aug)

        F_aug = A_aug * np.exp(1j * phi_c)

        return np.fft.irfft(F_aug, n=T)
