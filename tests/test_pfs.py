import pytest
import numpy as np
from mikeio import Mesh
from fmdap import Pfs


@pytest.fixture
def pfs():
    pfs_file = "tests/testdata/OresundHD2D_EnKF10/OresundHD2D_EnKF10.m21fm"
    return Pfs(pfs_file)


@pytest.fixture
def mesh():
    return Mesh("tests/testdata/Oresund_mesh_GEO.mesh")


def test_dda(pfs):
    assert "METHOD" in pfs.dda
    assert pfs.dda["METHOD"]["type"] == 1


def test_sections(pfs):
    assert "METHOD" in pfs.sections


def test_get_item(pfs):
    assert "type" in pfs["METHOD"]
    assert "type" in pfs["method"]


def test_get_attr(pfs):
    assert "type" in pfs.METHOD
    assert "type" in pfs.method


def test_model_errors(pfs):
    df = pfs.model_errors
    assert "include" in df
    assert len(df) == 2


def test_measurements(pfs):
    df = pfs.measurements
    assert len(df) == 4


def test_measurement_positions(pfs):
    df = pfs.measurement_positions
    assert "x" in df
    assert "name" in df
    assert len(df) == 4


def test_validate_positions(mesh, pfs):
    df = pfs.validate_positions(mesh, pfs.measurements)
    assert len(df) == 4


def test_diagnostics(pfs):
    df = pfs.diagnostics
    assert len(df) == 9
    assert df.loc[9].file_name == "Diagnostics_Global_DA_statistics.dfs0"
    assert np.all(df.type < 4)
