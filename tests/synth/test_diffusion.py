import numpy as np
import pandas as pd
import pytest

from metaforecast.synth.generators.diffusion import Diffusion, GaussianDiffusion


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "unique_id": ["A"] * 10,
            "ds": pd.date_range(start="2020-01-01", periods=10, freq="D"),
            "y": np.random.normal(0, 1, 10),
        }
    )


@pytest.fixture(params=[0.1, 0.2, 0.3])
def sigma(request):
    return request.param


def test_gaussian_diffusion(df):
    diffusion = GaussianDiffusion(sigma=0.2, knot=4, rename_uids=True)
    synth_df = diffusion.transform(df, n_series=1)

    assert synth_df.shape[0] == 10
    assert synth_df["unique_id"].nunique() == 1


def test_diffusion_generator(sigma, df):
    diffusion_model = Diffusion(sigma=sigma, knot=4, rename_uids=True)
    diffusion_model.train(df)
    synth_df = diffusion_model.transform(df, n_series=1)
    assert synth_df.shape[0] == 10
    assert synth_df["unique_id"].nunique() == 1
