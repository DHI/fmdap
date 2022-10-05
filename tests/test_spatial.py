import pytest
import numpy as np
import mikeio
from fmdap import spatial


@pytest.fixture
def dfs():
    wind_file = "tests/testdata/Wind_1hr.dfsu"
    return mikeio.open(wind_file)

@pytest.fixture
def dfs_nan():
    wind_file = "tests/testdata/Wind_1hr_nans.dfsu"
    return mikeio.open(wind_file)

def test_pairwise_distance():
    ec = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    d = spatial._pairwise_distance(ec)
    assert d.ndim == 2
    assert d.shape == (3, 3)
    assert np.all(d >= 0)
    assert d[1, 0] == 1.0


def test_get_distance_and_corrcoef(dfs,dfs_nan):
    d, cc = spatial.get_distance_and_corrcoef(dfs, n_sample=20)
    assert len(d) == 190
    assert np.all(d > 0)
    assert np.all(cc > 0)
    #test dfsu with nans and make sure it grabs a sample that comprises nans
    d, cc = spatial.get_distance_and_corrcoef(dfs_nan, n_sample=950)
    assert len(d) == 433846
    assert np.all(d > 0)
    assert np.all(cc > 0)


def test_gaussian():
    assert spatial.gaussian(0, 10) == 1.0
    assert spatial.gaussian(1e6, 10) < 1.0e-16
    res = spatial.gaussian(10, 10)
    assert pytest.approx(res) == 0.60653066


def test_fit_gaussian():
    dd = np.array([1, 3, 5, 8, 9, 12])
    vals = np.array([0.9, 0.85, 0.75, 0.5, 0.3, 0.22])
    sl = spatial.fit_gaussian(dd, vals)
    assert pytest.approx(sl) == 6.4027387
