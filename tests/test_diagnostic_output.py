import re
import sys, os
from fmdap import read_diagnostic
from fmdap.diagnostic_output import DiagnosticType
import pytest

# common
filename1 = "tests/testdata/Diagnostics_F16_EnKF.dfs0"
filename2 = "tests/testdata/diagnostics_nonMeas_SSC1.dfs0"
filename3 = "tests/testdata/Global_stats.dfs0"
filename1_alti = "Diagnostics_Altimetry_C2.dfs0"
filename1_OI = "Diagnostics_Drogden_OI.dfs0"


def test_folder():
    assert os.path.exists(filename1)


def test_read():
    diag = read_diagnostic(filename1)
    assert diag.df.values[0, 0] == 1.749464750289917
    assert diag.df.shape == (744, 12)


def test_diag_type_1():
    diag = read_diagnostic(filename1)
    assert diag.type == DiagnosticType.Measurement


def test_diag_type_1():
    diag = read_diagnostic(filename1)
    assert diag.type == DiagnosticType.Measurement


def test_diag_type_3():
    diag = read_diagnostic(filename3)
    assert diag.type == DiagnosticType.GlobalAssimilationStatistics


def test_get_total_forecast():
    diag = read_diagnostic(filename1)
    dff = diag.forecast.df
    assert dff.values[-1, 4] == 5.08772611618042
    assert dff.shape == (397, 12)


def test_get_total_analysis():
    diag = read_diagnostic(filename1)
    dfa = diag.analysis.df
    assert dfa.values[-1, 4] == 5.099072456359863
    assert dfa.shape == (347, 12)


def test_idx_at_updates():
    diag = read_diagnostic(filename1)
    iforecast = diag.idx_forecast
    nupdates = len(iforecast[iforecast == True])
    assert len(iforecast) == 744
    assert nupdates == 347
    assert diag.idx_no_update[-1] == False


def test_get_iforecast_from_ianalysis():
    diag = read_diagnostic(filename1)
    ianalysis = diag.idx_analysis
    iforecast2 = diag.get_iforecast_from_ianalysis(ianalysis)

    assert diag.idx_forecast[-1] == iforecast2[-1]


def test_get_increments():
    diag = read_diagnostic(filename1)
    dfi = diag.increment.df
    assert dfi.values[3, 3] == 0.013921260833740234


def test_get_all_increments_as_array():
    diag = read_diagnostic(filename1)
    arr = diag.increment.values.ravel()
    assert len(arr) == 3470


def test_get_mean_increments():
    diag = read_diagnostic(filename1)
    ianalysis = diag.idx_analysis
    dfi = diag._get_mean_increments()
    assert dfi.values[3, 0] == 0.03600311279296875
