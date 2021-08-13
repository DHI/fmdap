import pytest
import numpy as np
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
