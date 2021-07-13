import pytest
from fmdap import DiagnosticCollection


@pytest.fixture
def fldr_and_pfs():
    fldr = "tests/testdata/OresundHD2D_Free1/"
    fn = fldr + "OresundHD2D_1week.m21fm"
    return fldr, fn


@pytest.fixture
def dc(fldr_and_pfs):
    fldr, fn = fldr_and_pfs
    return DiagnosticCollection.from_pfs(fn, folder=fldr)


def test_repr(dc):
    assert "Viken" in repr(dc)


def test_getitem(dc):
    assert "Viken" in repr(dc)

    assert dc[0].name == "Viken"
    assert dc["Viken"].name == "Viken"

    assert len(dc.names) == 12
    assert len(dc) == 12

    assert dc[0].name == "Viken"
    assert dc["Viken"].name == "Viken"

    assert len(dc.names) == 12
    assert len(dc) == 12
