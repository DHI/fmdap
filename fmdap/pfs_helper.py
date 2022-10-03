from collections import namedtuple
import pandas as pd
from mikeio import Pfs


def pfs2dict(pfs_file):
    # with open(pfs_file, encoding='iso8859-1') as f:
    #         pfs = f.read()
    # y = pfs2yaml(pfs)
    # d = pfs2yaml.pfs2dict(pfs)
    d = Pfs(pfs_file).data.to_dict()
    return d


def get_DA_dict(d):
    root = list(d.keys())[0]
    return d[root]["DATA_ASSIMILATION_MODULE"]


def get_DA_sections(d):
    return list(get_DA_dict(d).keys())


def get_measurements_df(dda):
    # dda = get_DA_dict(d)
    meas_sec = dda.get("MEASUREMENTS")
    if meas_sec is None:
        raise KeyError("'MEASUREMENTS' section could not be found in dictionary!")
    n_meas = int(meas_sec.get("number_of_independent_measurements", 0))

    raw = {}
    for j in range(1, n_meas + 1):
        raw[j] = meas_sec[f"MEASUREMENT_{j}"]
    return pd.DataFrame(raw).T


def get_diagnostics_df(dda):
    # dda = get_DA_dict(d)
    diag_sec = dda.get("DIAGNOSTICS", {}).get("OUTPUTS")
    if diag_sec is None:
        raise KeyError("DIAGNOSTICS/OUTPUTS section could not be found in dictionary!")
    n_diag = int(diag_sec.get("number_of_outputs", 0))

    raw = {}
    for j in range(1, n_diag + 1):
        raw[j] = diag_sec[f"OUTPUT_{j}"]
    return pd.DataFrame(raw).T
