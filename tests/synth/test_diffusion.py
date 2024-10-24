import numpy as np
import pandas as pd

from metaforecast.synth.generators.diffusion import Diffusion

def test_diffusion():
    df = pd.DataFrame({
        'unique_id': ['A'] * 10,
        'ds': pd.date_range(start='2020-01-01',
                            periods=10,
                            freq='D'),
        'y': np.random.normal(0, 1, 10)
    })

    diffusion = Diffusion(sigma=0.2, knot=4, rename_uids=True)
    synth_df = diffusion.transform(df, n_series=1)

    assert synth_df.shape[0] == 10
    assert synth_df['unique_id'].nunique() == 1
