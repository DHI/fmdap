import mikeio
import pandas as pd
from mikeio import PfsDocument


def pfs2dict(pfs_file):
    d = PfsDocument(pfs_file).to_dict()
    return d


def get_DA_dict(d):
    root = list(d.keys())[0]
    return d[root]["DATA_ASSIMILATION_MODULE"]


def get_DA_sections(d):
    return list(get_DA_dict(d).keys())


def get_measurements_df(dda: mikeio.PfsSection) -> pd.DataFrame:
    return dda.MEASUREMENTS.to_dataframe()


def get_diagnostics_df(dda: mikeio.PfsSection) -> pd.DataFrame:
    return dda.DIAGNOSTICS.OUTPUTS.to_dataframe()
