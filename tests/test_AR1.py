import pytest
import numpy as np
import pandas as pd
from mikeio import Dfs0
from fmdap import AR1


@pytest.fixture
def ts():
    fn = "tests/testdata/eq_ts_with_gaps.dfs0"
    return Dfs0(fn).to_dataframe()


def test_phi_to_halflife():
    phi = 0.95
    halflife = AR1.phi_to_halflife(phi)
    assert pytest.approx(halflife) == 13.513407

    dt = 1800
    halflife = AR1.phi_to_halflife(phi, dt=dt)
    assert pytest.approx(halflife) == 13.513407 * 1800


def test_halflife_to_phi():
    halflife = 13.513407333964901
    phi = AR1.halflife_to_phi(halflife)
    assert pytest.approx(phi) == 0.95

    dt = 1800
    phi = AR1.halflife_to_phi(halflife * 1800, dt=dt)
    assert pytest.approx(phi) == 0.95


def test_estimate_AR1_halflife(ts):
    halflife = AR1.estimate_AR1_halflife(ts) / 3600
    assert np.round(halflife, 3) == 93.657

    with pytest.raises(TypeError):
        # only DataFrame/Series accepted
        AR1.estimate_AR1_halflife(ts.to_numpy())


def test_simulate_AR1(ts):
    rho = 3 * 3600

    df = AR1.simulate_AR1(halflife=rho, index=ts.index)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(ts)

    phi = AR1.halflife_to_phi(rho)
    vals = AR1.simulate_AR1(phi=phi, n_sample=len(ts))
    assert isinstance(vals, np.ndarray)
    assert len(vals) == len(ts)

    with pytest.raises(ValueError):
        # must provide either phi or halflife
        AR1.simulate_AR1(n_sample=len(ts))
