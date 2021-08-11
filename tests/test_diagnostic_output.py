import re
import sys, os
import numpy as np
from fmdap import read_diagnostic
from fmdap import diagnostic_output as do
from mikeio import Dfs0
import pytest

# common
filename_EnKF = "tests/testdata/Diagnostics_F16_EnKF.dfs0"
filename_nonMeas = "tests/testdata/diagnostics_nonMeas_SSC1.dfs0"
filename_GlobalStats = "tests/testdata/Global_stats.dfs0"
filename_EnKF_alti = "tests/testdata/Diagnostics_Altimetry_C2.dfs0"
filename_OI = "tests/testdata/Diagnostics_Drogden_OI.dfs0"


def test_folder():
    assert os.path.exists(filename_EnKF)


def test_read():
    diag = read_diagnostic(filename_EnKF)
    assert diag.df.values[0, 0] == pytest.approx(1.7494647)
    assert diag.df.shape == (744, 12)


def test_read_as_dataframe():
    df = Dfs0(filename_EnKF).to_dataframe()
    diag = read_diagnostic(df)
    assert diag.df.values[0, 0] == pytest.approx(1.7494647)
    assert diag.df.shape == (744, 12)


def test_diag_type_1():
    diag = read_diagnostic(filename_EnKF)
    assert diag.type == do.DiagnosticType.Measurement
    assert diag.has_measurement
    assert diag.is_ensemble
    assert diag.n_members == 10
    assert diag.is_point
    assert diag.n_updates == 347
    assert diag.has_updates
    assert diag.values.shape == (744, 10)

    assert "Measurements: " in repr(diag)


def test_plot_diag_type_1():
    diag = read_diagnostic(filename_EnKF)

    diag.scatter()
    diag.ecdf()
    diag.hist()


def test_diag_type_2():
    diag = read_diagnostic(filename_nonMeas)
    assert diag.type == do.DiagnosticType.NonMeasurementPoint
    assert diag.n_members == 3
    assert diag.n == 114
    assert len(diag) == 114
    assert diag.has_measurement == False
    assert diag.is_ensemble
    assert diag.is_point
    assert diag.n_updates == 0
    assert diag.values.shape == (114, 3)

    assert "Measurements: " not in repr(diag)


def test_diag_type_3():
    diag = read_diagnostic(filename_GlobalStats)
    assert diag.type == do.DiagnosticType.GlobalAssimilationStatistics


def test_get_total_forecast():
    diag = read_diagnostic(filename_EnKF)
    dff = diag.forecast.df
    assert dff.values[-1, 4] == pytest.approx(5.087726)
    assert dff.shape == (397, 12)


def test_get_total_analysis():
    diag = read_diagnostic(filename_EnKF)
    dfa = diag.analysis.df
    assert dfa.values[-1, 4] == pytest.approx(5.0990724)
    assert dfa.shape == (347, 12)


def test_idx_at_updates():
    diag = read_diagnostic(filename_EnKF)
    iforecast = diag.idx_forecast
    nupdates = len(iforecast[iforecast == True])
    assert len(iforecast) == 744
    assert nupdates == 347
    assert diag.idx_no_update[-1] == False


def test_get_increments():
    diag = read_diagnostic(filename_EnKF)
    dfi = diag.increment.df
    assert dfi.values[3, 3] == pytest.approx(0.01392126)


def test_get_all_increments_as_array():
    diag = read_diagnostic(filename_EnKF)
    arr = diag.increment.values.ravel()
    assert len(arr) == 3470


def test_get_mean_increments():
    diag = read_diagnostic(filename_EnKF)
    ianalysis = diag.idx_analysis
    dfi = diag._get_mean_increments()
    assert dfi.values[3, 0] == pytest.approx(0.03600311)


# ======= forecast ==========


def test_forecast_EnKF():
    diag = read_diagnostic(filename_EnKF)
    res = diag.forecast
    assert isinstance(res, do.DiagnosticResults)
    assert res.n_members == 10
    assert res.values.shape == (397, 10)

    res.ecdf()


def test_forecast_nonMeas():
    diag = read_diagnostic(filename_nonMeas)
    res = diag.forecast
    assert isinstance(res, do.DiagnosticResults)
    assert len(res) == 114
    assert len(np.unique(res.time)) == 114
    assert res.n_members == 3
    assert res.values.shape == (114, 3)


def test_forecast_EnKF_alti():
    diag = read_diagnostic(filename_EnKF_alti)
    res = diag.forecast
    assert isinstance(res, do.DiagnosticResults)
    assert len(res) == 31
    assert len(np.unique(res.time)) == 1
    assert res.n_members == 7
    assert res.values.shape == (31, 7)

    res.plot()


def test_forecast_OI():
    diag = read_diagnostic(filename_OI)
    res = diag.forecast
    assert isinstance(res, do.DiagnosticResults)
    assert len(res) == 2017
    assert len(np.unique(res.time)) == 2017
    assert res.n_members == 1
    assert res.values.shape == (2017, 1)

    res.hist()


# ======= forecast_at_update ==========


def test_forecast_at_update_EnKF():
    diag = read_diagnostic(filename_EnKF)
    res = diag.forecast_at_update
    assert isinstance(res, do.DiagnosticResults)
    assert res.n_members == 10
    assert res.values.shape == (347, 10)

    res.ecdf()


def test_forecast_at_update_nonMeas():
    diag = read_diagnostic(filename_nonMeas)
    assert diag.forecast_at_update is None


def test_forecast_at_update_EnKF_alti():
    diag = read_diagnostic(filename_EnKF_alti)
    res = diag.forecast_at_update
    assert isinstance(res, do.DiagnosticResults)
    assert len(res) == 31
    assert len(np.unique(res.time)) == 1
    assert res.n_members == 7
    assert res.values.shape == (31, 7)

    res.plot()


def test_forecast_at_update_OI():
    diag = read_diagnostic(filename_OI)
    res = diag.forecast_at_update
    assert isinstance(res, do.DiagnosticResults)
    assert len(res) == 168
    assert len(np.unique(res.time)) == 168
    assert res.n_members == 1
    assert res.values.shape == (168, 1)

    res.hist()


# ======= analysis ==========


def test_analysis_EnKF():
    diag = read_diagnostic(filename_EnKF)
    res = diag.analysis
    assert isinstance(res, do.DiagnosticResults)
    assert res.n_members == 10
    assert res.values.shape == (347, 10)

    assert res.ensemble.shape == (347, 10)

    mi = res.min()
    ma = res.max()

    res.plot()


def test_analysis_nonMeas():
    diag = read_diagnostic(filename_nonMeas)
    assert diag.analysis is None


def test_analysis_EnKF_alti():
    diag = read_diagnostic(filename_EnKF_alti)
    res = diag.analysis
    assert isinstance(res, do.DiagnosticResults)
    assert len(res) == 31
    assert len(np.unique(res.time)) == 1
    assert res.n_members == 7
    assert res.values.shape == (31, 7)

    assert np.all(res.max() - res.min() >= 0)
    res.hist()


def test_analysis_OI():
    diag = read_diagnostic(filename_OI)
    res = diag.analysis
    assert isinstance(res, do.DiagnosticResults)
    assert len(res) == 168
    assert len(np.unique(res.time)) == 168
    assert res.n_members == 1
    assert res.values.shape == (168, 1)

    res.ecdf()
    res.scatter()


# ======= result ==========


def test_result_EnKF():
    diag = read_diagnostic(filename_EnKF)
    res = diag.result
    assert isinstance(res, do.DiagnosticResults)
    assert res.n_members == 10
    assert res.values.shape == (397, 10)

    assert np.all(res.median() - res.min() >= 0)

    res.hist()


def test_result_nonMeas():
    diag = read_diagnostic(filename_nonMeas)
    res = diag.result
    assert isinstance(res, do.DiagnosticResults)
    assert len(res) == 114
    assert len(np.unique(res.time)) == 114
    assert res.n_members == 3
    assert res.values.shape == (114, 3)


def test_result_EnKF_alti():
    diag = read_diagnostic(filename_EnKF_alti)
    res = diag.result
    assert isinstance(res, do.DiagnosticResults)
    assert len(res) == 31
    assert len(np.unique(res.time)) == 1
    assert res.n_members == 7
    assert res.values.shape == (31, 7)

    res.ecdf()


def test_result_OI():
    diag = read_diagnostic(filename_OI)
    res = diag.result
    assert isinstance(res, do.DiagnosticResults)
    assert len(res) == 2017
    assert len(np.unique(res.time)) == 2017
    assert res.n_members == 1
    assert res.values.shape == (2017, 1)

    assert res.std().sum() == 0

    res.plot()


# ======= innovation ==========


def test_innovation_EnKF():
    diag = read_diagnostic(filename_EnKF)
    inno = diag.innovation
    assert isinstance(inno, do.DiagnosticInnovations)
    assert len(inno) == 125
    assert inno.n_members == 10
    assert inno.values.shape == (125, 10)

    assert np.all(inno.mean() - inno.min() >= 0)

    inno.hist()


def test_innovation_EnKF_alti():
    diag = read_diagnostic(filename_EnKF_alti)
    inno = diag.innovation
    assert isinstance(inno, do.DiagnosticInnovations)
    assert len(inno) == 62
    assert len(np.unique(inno.time)) == 1
    assert inno.n_members == 7
    assert inno.values.shape == (62, 7)

    inno.ecdf()


def test_innovation_OI():
    diag = read_diagnostic(filename_OI)
    inno = diag.innovation
    assert isinstance(inno, do.DiagnosticInnovations)
    assert len(inno) == 1171
    assert len(np.unique(inno.time)) == 1004
    assert inno.n_members == 1
    assert inno.shape == (1171, 1)

    inno.plot()


# ======= increment ==========


def test_increment_EnKF():
    diag = read_diagnostic(filename_EnKF)
    incr = diag.increment
    assert isinstance(incr, do.DiagnosticIncrements)
    assert incr.n_members == 10
    assert incr.values.shape == (347, 10)

    incr.ecdf()


def test_increment_nonMeas():
    diag = read_diagnostic(filename_nonMeas)
    incr = diag.increment
    assert incr is None


def test_increment_EnKF_alti():
    diag = read_diagnostic(filename_EnKF_alti)
    incr = diag.increment
    assert isinstance(incr, do.DiagnosticIncrements)
    assert len(incr) == 31
    assert len(np.unique(incr.time)) == 1
    assert incr.n_members == 7
    assert incr.values.shape == (31, 7)

    incr.plot()


def test_increment_OI():
    diag = read_diagnostic(filename_OI)
    incr = diag.increment
    assert isinstance(incr, do.DiagnosticIncrements)
    assert len(incr) == 168
    assert len(np.unique(incr.time)) == 168
    assert incr.n_members == 1
    assert incr.values.shape == (168, 1)

    incr.hist()


# ======= skill ==========


def test_skill_EnKF():
    diag = read_diagnostic(filename_EnKF, name="F16")
    s = diag.skill()
    assert s["rmse"]["F16 analysis"] == pytest.approx(0.4192413)
    assert s["rmse"]["F16 forecast"] == pytest.approx(0.3935569)

    sa = diag.analysis.skill()
    assert sa["rmse"]["F16 analysis"] == s["rmse"]["F16 analysis"]

    sf = diag.forecast.skill()
    assert sf["rmse"]["F16 forecast"] == s["rmse"]["F16 forecast"]


def test_skill_EnKF_alti():
    diag = read_diagnostic(filename_EnKF_alti, name="alti")
    s = diag.skill()

    sa = diag.analysis.skill()
    assert sa["rmse"]["alti analysis"] == s["rmse"]["alti analysis"]

    sf = diag.forecast.skill()
    assert sf["rmse"]["alti forecast"] == s["rmse"]["alti forecast"]

