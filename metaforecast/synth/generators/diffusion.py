import numpy as np
import pandas as pd
import torch

from metaforecast.synth.generators.base import SemiSyntheticGenerator


class GaussianDiffusion(SemiSyntheticGenerator):
    """Generate synthetic time series using diffusion.

    Forward phase referred to as diffusion.
    Injects Gaussian noise across a set of training samples.

    References
    ----------
    Yang, Yiyuan, et al. (2024) "A survey on diffusion models for time
    series and spatio-temporal data."
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
        super().__init__(alias="GaussianDiffusion")

        self.sigma = sigma
        self.knot = knot
        self.rename_uids = rename_uids

    def transform(self, df: pd.DataFrame, n_series: int = -1, **kwargs):
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

        if n_series < 0:
            n_series = df["unique_id"].nunique()

        ts_list = []
        for _ in range(n_series):
            uid = (
                f"Diffusion_{self.counter}"
                if self.rename_uids
                else df["unique_id"].sample(1).values[0]
            )
            ts = self._create_synthetic_ts(df)
            ts["unique_id"] = uid
            ts_list.append(ts)
            self.counter += 1

        dataset = pd.concat(ts_list)

        return dataset

    # pylint: disable=arguments-differ
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
        df_["y"] = self._apply_diffusion(df_["y"].values)
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
        random_warps = np.random.normal(
            loc=1.0, scale=self.sigma, size=(self.knot + 2, x.shape[1])
        )
        # Computes evenly spaced knots for the diffusion path.
        warp_steps = np.linspace(0, x.shape[0] - 1.0, num=self.knot + 2)
        time_warp = np.zeros((x.shape[0], x.shape[1]))
        x_warped = np.zeros((x.shape[0], x.shape[1]))

        for i in range(x.shape[1]):
            # Finds the new time steps after applying the diffusion path.
            time_warp[:, i] = np.interp(
                orig_steps, warp_steps, warp_steps * random_warps[:, i]
            )
            # Finds the new values after applying the diffusion path.
            x_warped[:, i] = np.interp(time_warp[:, i], orig_steps, x[:, i])

        return x_warped.squeeze()


class ExampleDiffusionModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x


class Diffusion(SemiSyntheticGenerator):
    """

    Implements diffusion model that undergoes a two-step process involving the injection of
    Gaussian noise and the subsequent removal of that noise. By training the model to predict
    the noise added during the diffusion process, the models learns to generate synthetic
    time series samples.

    References
    ----------
    Yang, Yiyuan, et al. (2024) "A survey on diffusion models for time series and
    spatio-temporal data."
    """

    def __init__(self, sigma: float = 0.2, knot=4, rename_uids: bool = True):
        """Initialize diffusion model generator with parameters.

        Parameters
        ----------
        sigma : float
            Standard deviation of the Gaussian noise added to the time series.
        knot : int
            Number of knots used to generate the diffusion path.
        rename_uids : bool
            Whether to rename the unique identifiers of the synthetic series.

        """
        super().__init__(alias="Diffusion")
        self.sigma = sigma
        self.knot = knot
        self.rename_uids = rename_uids
        self.gaussian_diffusion = GaussianDiffusion(sigma, knot, rename_uids)

    def train(
            self,
            df: pd.DataFrame,
            epochs=1,
            learning_rate=0.01,
            diffusion_model=None,
    ):
        """Train the diffusion model.

        Parameters
        ----------
        df : pd.DataFrame
            Time series dataset with required columns:
            - unique_id: Series identifier
            - ds: Timestamp
            - y: Target values
        epochs : int
            Number of training epochs.
        learning_rate : float
            Learning rate for the optimizer.
        diffusion_model : torch.nn.Module
            Diffusion model to train.
        """
        if not diffusion_model:
            diffusion_model = ExampleDiffusionModel(df.shape[0], df.shape[0])
        # pylint: disable=attribute-defined-outside-init
        # TODO(Rui/Vítor): Once we better define trainable models, we can remove this pylint.
        self.model = diffusion_model
        for _ in range(epochs):
            synthetic_df = self.gaussian_diffusion.transform(df, 1)
            real_noise = synthetic_df["y"].values - df["y"].values

            predicted_noise = self.model(
                torch.tensor(synthetic_df["y"].values, dtype=torch.float32).unsqueeze(1)
            )

            loss = torch.nn.functional.mse_loss(
                predicted_noise,
                torch.tensor(real_noise, dtype=torch.float32).unsqueeze(1),
            )

            self.model.zero_grad()
            loss.backward()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            optimizer.step()

    def transform(self, df: pd.DataFrame, n_series: int, **kwargs):
        """Generate synthetic time series using diffusion model.

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
            - y values generated using diffusion model

        """
        self._assert_datatypes(df)
        self._assert_model_trained()

        dataset = []
        for _ in range(n_series):
            uid = (
                f"Diffusion_{self.counter}"
                if self.rename_uids
                else df["unique_id"].sample(1).values[0]
            )
            ts = self._create_synthetic_ts(df)
            ts["unique_id"] = uid
            dataset.append(ts)
            self.counter += 1

        return pd.concat(dataset)

    def _assert_model_trained(self):
        """
        Assert that the diffusion model has been trained.

        Raises
        ------
        ValueError
            If the model has not been trained.

        """
        if not hasattr(self, "model"):
            raise ValueError(
                """Diffusion model must be trained before generating synthetic series.
            Use the `train` method to train the model."""
            )

    # pylint: disable=arguments-differ
    def _create_synthetic_ts(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Apply diffusion model to a time series.

        Parameters
        ----------
        x : np.ndarray
            Time series values.

        Returns
        -------
        np.ndarray
            Time series values after applying diffusion model.

        """

        # Receives a noisy time series and predicts the noise to be removed.
        x = df["y"].values
        x = x.reshape(-1, 1)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        predicted_noise = self.model(x)
        x = x.squeeze()
        predicted_noise = predicted_noise.squeeze().detach().numpy()
        x = x.numpy()
        x = x - predicted_noise
        return pd.DataFrame({"ds": df["ds"], "y": x})
