import pytest
import numpy as np
from fmdap import DiagnosticCollection


@pytest.fixture
def fldr_and_pfs():
    fldr = "tests/testdata/OresundHD2D_Free1/"
    fn = fldr + "OresundHD2D_1week.m21fm"
    return fldr, fn


@pytest.fixture
def dc_free1(fldr_and_pfs):
    fldr, fn = fldr_and_pfs
    return DiagnosticCollection.from_pfs(fn, folder=fldr)


@pytest.fixture
def fldr_and_pfs_EnKF10():
    fldr = "tests/testdata/OresundHD2D_EnKF10/"
    fn = fldr + "OresundHD2D_EnKF10.m21fm"
    return fldr, fn


@pytest.fixture
def dc_EnKF10(fldr_and_pfs_EnKF10):
    fldr, fn = fldr_and_pfs_EnKF10
    return DiagnosticCollection.from_pfs(fn, folder=fldr)


def test_repr(dc_free1):
    assert "Viken" in repr(dc_free1)


def test_getitem_free(dc_free1):
    dc = dc_free1
    assert "Viken" in repr(dc)

    assert dc[0].name == "Viken"
    assert dc["Viken"].name == "Viken"

    assert len(dc.names) == 12
    assert len(dc) == 12


def test_getitem_EnKF(dc_EnKF10):
    dc = dc_EnKF10
    assert "Viken" in repr(dc)

    assert dc[0].name == "Viken"
    assert dc["Viken"].name == "Viken"

    assert len(dc.names) == 6
    assert len(dc) == 6


def test_forecast_at_update(dc_EnKF10):
    fc = dc_EnKF10.forecast_at_update
    assert len(fc) == 6
    assert fc["Klagshamn"].n == 1105


def test_increment(dc_EnKF10):
    incr = dc_EnKF10.increment
    assert len(incr) == 6
    assert incr["Klagshamn"].n == 1105


def test_innovation(dc_EnKF10):
    inno = dc_EnKF10.innovation
    assert len(inno) == 4
    assert inno["Viken"].n == 2258


def test_add_diagnostic(dc_EnKF10):
    dc = dc_EnKF10
    assert len(dc) == 6

    d1 = dc["Drogden"].copy()
    d1.name = "New_drogden"
    dc.add_diagnostics(d1)
    assert len(dc) == 7
    assert "New_drogden" in dc

    d2 = d1
    dc.add_diagnostics(d2, names="Drogden2")
    assert len(dc) == 8
    assert "Drogden2" in dc


def test_sel(dc_EnKF10):
    dc = dc_EnKF10

    dcs = dc.sel(measured_variable="water level")
    assert len(dcs) == 4

    dcs = dc.sel(type=2)
    assert len(dcs) == 2
    assert "Diagnostics_wlbc_err_North" in dcs

    dcs = dc.sel(assimilated=True)
    assert len(dcs) == 2
    assert "Viken" in dcs


def test_query(dc_EnKF10):
    dc = dc_EnKF10
    dcq = dc.query("measurement_id<4 & data_offset>0.2")
    assert "Viken" in dcq


def test_skill(dc_EnKF10):
    dc = dc_EnKF10
    s = dc.skill().to_dataframe()
    assert s.loc["Barsebaeck", "result"].rmse == pytest.approx(0.0351065)
    assert s.loc["Barsebaeck", "analysis"].bias == pytest.approx(-0.021444086)
    assert s.loc["Klagshamn", "forecast"].n == 97

    sr = dc.result.skill()
    assert sr.loc["Barsebaeck"].rmse == s.loc["Barsebaeck", "result"].rmse

    sa = dc.analysis.skill()
    assert sa.loc["Barsebaeck"].bias == s.loc["Barsebaeck", "analysis"].bias

    sf = dc.forecast.skill()
    assert sf.loc["Klagshamn"].n == s.loc["Klagshamn", "forecast"].n


def test_scatter(dc_EnKF10):
    dc_EnKF10.scatter()


def test_rmse(dc_EnKF10):
    dc = dc_EnKF10
    assert len(dc.rmse) == 6
    assert np.all(dc.rmse.loc["Diagnostics_wlbc_err_North"].isnull())


def test_bias(dc_EnKF10):
    dc = dc_EnKF10
    assert len(dc.bias) == 6
    assert dc.bias.loc["Drogden"].iloc[0] == pytest.approx(-0.005636754)


def test_ensemble_std(dc_EnKF10):
    dc = dc_EnKF10
    assert len(dc.ensemble_std) == 6
    assert np.all(dc.ensemble_std > 0)
    assert dc.ensemble_std.loc["Viken"].iloc[0] == pytest.approx(0.01299819)
