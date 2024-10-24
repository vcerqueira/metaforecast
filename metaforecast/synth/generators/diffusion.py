import numpy as np
import pandas as pd

from metaforecast.synth.generators.base import SemiSyntheticGenerator


class Diffusion(SemiSyntheticGenerator):
    """Generate synthetic time series using diffusion.

    Incoporates Gaussian noise to the time series to generate synthetic data.
    The diffusion path is created by adding Gaussian noise to the time series.
    """

    def __init__(self, sigma: float = 0.2, knot=4, rename_uids: bool = True):
        """Initialize diffusion generator with parameters.

        Parameters
        ----------
        sigma : float
            Standard deviation of the Gaussian noise added to the time series.
        knot : int
            Number of knots used to generate the diffusion path.
        rename_uids : bool
            Whether to rename the unique identifiers of the synthetic series.

        """
        super().__init__(alias='Diffusion')
        self.sigma = sigma
        self.knot = knot
        self.rename_uids = rename_uids

    def transform(self, df: pd.DataFrame, n_series: int, **kwargs):
        """Generate synthetic time series using diffusion.

        Parameters
        ----------
        df : pd.DataFrame
            Source time series dataset with required columns:
            - unique_id: Series identifier
            - ds: Timestamp
            - y: Target values
            Must follow nixtla framework conventions
        n_series : int
            Number of synthetic series to generate.
        kwargs : dict
            Additional keyword arguments.
            
        Returns 
        -------
        pd.DataFrame
            Generated synthetic series with the same structure:
            - New unique_ids: f"Diffusion_{i}" for i in range(n_series)
            - Same temporal alignment as the source
            - y values generated using diffusion

        """
        self._assert_datatypes(df)

        dataset = []
        for _ in range(n_series):
            uid = f'Diffusion_{self.counter}' if self.rename_uids else df['unique_id'].sample(1).values[0]
            ts = self._create_synthetic_ts(df)
            ts['unique_id'] = uid
            dataset.append(ts)
            self.counter += 1

        return pd.concat(dataset)
    
    def _create_synthetic_ts(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Apply diffusion to a time series dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Time series dataset with required columns:
            - unique_id: Series identifier
            - ds: Timestamp
            - y: Target values
        kwargs : dict
            Additional keyword arguments.
        
        Returns
        -------
        pd.DataFrame
            Time series dataset with the same structure:
            - Original columns preserved
            - Same temporal alignment
            - Modified y values

        """
        df_ = df.copy()
        df_['y'] = self._apply_diffusion(df_['y'].values)
        return df_
    
    def _apply_diffusion(self, x: np.ndarray) -> np.ndarray:
        """
        Apply diffusion to a time series.

        Parameters
        ----------
        x : np.ndarray
            Time series values.

        Returns
        -------
        np.ndarray
            Time series values after applying diffusion.

        """
        x = x.reshape(-1, 1)
        orig_steps = np.arange(x.shape[0])
        # Adds 2 extra knots to the diffusion path for boundary conditions: start and end.
        random_warps = np.random.normal(loc=1.0, scale=self.sigma, size=(self.knot + 2, x.shape[1]))
        # Computes evenly spaced knots for the diffusion path.
        warp_steps = np.linspace(0, x.shape[0] - 1.0, num=self.knot + 2)
        time_warp = np.zeros((x.shape[0], x.shape[1]))
        x_warped = np.zeros((x.shape[0], x.shape[1]))

        for i in range(x.shape[1]):
            # Finds the new time steps after applying the diffusion path.
            time_warp[:, i] = np.interp(orig_steps, warp_steps, warp_steps * random_warps[:, i])
            # Finds the new values after applying the diffusion path.
            x_warped[:, i] = np.interp(time_warp[:, i], orig_steps, x[:, i])

        return x_warped.squeeze()
    
