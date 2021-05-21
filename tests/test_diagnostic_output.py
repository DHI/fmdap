import re
import sys, os
from fmdap import diagnostic_output
import pytest

# common 
filename1 = "tests/testdata/Diagnostics_F16.dfs0"
filename3 = 'tests/testdata/Global_stats.dfs0'

def test_folder():
    assert os.path.exists(filename1)

def test_read():
    diag = diagnostic_output.DiagnosticOutput()
    diag.read(filename1)
    assert diag.df.values[0,0] == 1.749464750289917
    assert diag.df.shape == (744, 12)


def test_diag_type_1():
    diag = diagnostic_output.DiagnosticOutput()
    diag.read(filename1)    
    assert diag.diag_type == 1


def test_diag_type_3():
    diag = diagnostic_output.DiagnosticOutput()
    diag.read(filename3)    
    assert diag.diag_type == 3


def test_get_total_forecast():
    diag = diagnostic_output.DiagnosticOutput()
    diag.read(filename1)
    dff = diag.get_total_forecast()    
    assert dff.values[-1,4] == 5.08772611618042
    assert dff.shape == (397, 12)


def test_get_total_analysis():
    diag = diagnostic_output.DiagnosticOutput()
    diag.read(filename1)
    dfa = diag.get_total_analysis()    
    assert dfa.values[-1,4] == 5.099072456359863
    assert dfa.shape == (397, 12) 


def test_idx_at_updates():
    diag = diagnostic_output.DiagnosticOutput()
    diag.read(filename1)
    iforecast, ianalysis, inoupdate = diag.idx_at_updates()
    nupdates = len(iforecast[iforecast==True])
    assert len(iforecast) == 744
    assert nupdates == 347
    assert inoupdate[-1] == False


def test_get_iforecast_from_ianalysis():
    diag = diagnostic_output.DiagnosticOutput()
    diag.read(filename1)
    iforecast, ianalysis, inoupdate = diag.idx_at_updates()
    iforecast2 = diag.get_iforecast_from_ianalysis(ianalysis)

    assert iforecast[-1] == iforecast2[-1]


def test_get_increments():
    diag = diagnostic_output.DiagnosticOutput()
    diag.read(filename1)    
    dfi = diag.get_increments()
    assert dfi.values[3,3] == 0.013921260833740234


def test_get_all_increments_as_array():
    diag = diagnostic_output.DiagnosticOutput()
    diag.read(filename1)   
    arr = diag.get_all_increments_as_array() 
    assert len(arr) == 3470


def test_get_mean_increments():
    diag = diagnostic_output.DiagnosticOutput()
    diag.read(filename1)    
    _, ianalysis, _ = diag.idx_at_updates()
    dfi = diag.get_mean_increments()
    assert dfi.values[3,0] == 0.03600311279296875
