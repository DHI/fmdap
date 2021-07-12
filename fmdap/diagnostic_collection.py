from collections.abc import Mapping, Iterable
import os
import numpy as np
import warnings
import pandas as pd
from fmdap.diagnostic_output import DiagnosticDataframe, DiagnosticType, read_diagnostic
from fmdap import pfs_helper as pfs


class DiagnosticCollection(Mapping):
    @property
    def df_attrs(self):
        if self._df_attrs is None:
            all_attrs = [self.diagnostics[n].attrs for n in self.names]
            self._df_attrs = pd.DataFrame(all_attrs, index=self.names)
        return self._df_attrs

    def __init__(self, diagnostics=None, names=None, attrs=None):
        self.diagnostics = {}
        self.names = []
        self._df_attrs = None
        if diagnostics is not None:
            self.add_diagnostics(diagnostics, names, attrs)

    @classmethod
    def from_pfs(cls, pfs_file, folder=None, types=None):
        df, DA_type = cls._parse_pfs(pfs_file, types)
        df = cls._check_file_existance(df, folder)
        dc = cls()
        for _, row in df.iterrows():
            name = row["name"] if (not pd.isnull(row["name"])) else None
            attrs = row.dropna().to_dict()
            attrs["DA_type"] = DA_type
            attrs["pfs_file"] = pfs_file
            dc._add_single_diagnostics(row["file_name"], name=name, attrs=attrs)
        return dc

    @classmethod
    def _parse_pfs(cls, pfs_file, types=None):

        warnings.filterwarnings("ignore", message="Support for PFS files")
        assert os.path.exists(pfs_file)
        d = pfs.pfs2dict(pfs_file).get("DATA_ASSIMILATION_MODULE")
        if d is None:
            raise ValueError(
                "'DATA_ASSIMILATION_MODULE' section could not be found in pfs file!"
            )
        DA_type = d.get("METHOD", {}).get("type", 0)

        dfd = pfs.get_diagnostics_df(d)
        if "include" not in dfd:
            dfd["include"] = 1
        dfd.index.name = "output_id"
        if dfd.type.isin([1]).any():
            dfm = pfs.get_measurements_df(d)
            if "include" not in dfm:
                dfm["include"] = 1
            df = dfd.join(dfm, on="measurement_id", lsuffix="", rsuffix="_measurement")
            if "include_measurement" in df:
                df.loc[df.include.isnull(), "include_measurement"] = 1
                if DA_type == 0:
                    df["assimilated"] = False
                else:
                    df["assimilated"] = df["include_measurement"] == 1
        else:
            df = dfd

        if types is not None:
            types = [types] if isinstance(types, int) else types
            df = df[df.type.isin(types)]

        if "include" in df:
            df.loc[df.include.isnull(), "include"] = 1
            df = df[df.include == 1].drop(columns="include")

        return df.dropna(axis=1, how="all"), DA_type

    @classmethod
    def _check_file_existance(cls, df, folder=None):
        if folder is None:
            folder = ""

        file_name = np.array([os.path.join(folder, x) for x in df["file_name"]])
        file_exists = np.array([os.path.exists(x) for x in file_name])
        if "file_name_measurement" in df:
            measurement_file_exists = np.array(
                [
                    os.path.exists(os.path.join(folder, x))
                    for x in df["file_name_measurement"]
                    if (not pd.isnull(x))
                ]
            )
            df["measurement_file_exists"] = False
            df.loc[
                ~pd.isnull(df["file_name_measurement"]), "measurement_file_exists"
            ] = measurement_file_exists

        if not any(file_exists):
            raise ValueError(
                f"None of the diagnostic files exists in the given folder '{folder}'. Use the 'folder' argument to adjust the paths."
            )
        df["file_name"] = file_name

        return df[file_exists == True]

    def add_diagnostics(self, diagnostics, names=None, attrs=None):
        # TODO: take single file, folder, files with wildcard
        # pfs/log file or DiagnosticDataframe
        if isinstance(diagnostics, str) or (not isinstance(diagnostics, Iterable)):
            diagnostics = [diagnostics]
            if names is not None:
                names = [names]

        if names is None:
            names = [None]
        else:
            if len(names) != len(diagnostics):
                raise ValueError("diagnostics and names must have same length!")
        if attrs is None:
            attrs = [None]
        # else:
        #     if len(attrs) != len(diagnostics):
        #         raise ValueError("attrs must have same length as diagnostics!")

        for name, diag, attr in zip(names, diagnostics, attrs):
            self._add_single_diagnostics(diag, name=name, attrs=attr)

    def _add_single_diagnostics(self, diagnostic, name=None, attrs=None):
        if isinstance(diagnostic, str):
            diagnostic = read_diagnostic(diagnostic, name=name)
        assert isinstance(diagnostic, DiagnosticDataframe)
        if name is None:
            name = diagnostic.name
        if name in self.names:
            raise ValueError(
                f"Cannot add diagnostic with name={name}; already in {self.names}"
            )
        if attrs:
            diagnostic.attrs = attrs
        self.names.append(name)
        self.diagnostics[name] = diagnostic
        self._df_attrs = None

    def __repr__(self):
        out = []
        out.append(f"<{type(self).__name__}>")
        for name in self.names:
            diag = self[name]
            out.append(
                f" - {name}: type:{diag.type.name}, n:{diag.n}; {type(diag).__name__}"
            )
        return str.join("\n", out)

    def __getitem__(self, x):
        if isinstance(x, int):
            x = self._get_diag_name(x)

        return self.diagnostics[x]

    def _get_diag_name(self, diag):
        return self.names[self._get_diag_id(diag)]

    def _get_diag_id(self, diag):
        # if diag is None:  # or len(self) <= 1:
        #    return 0
        if isinstance(diag, str):
            if diag in self.names:
                diag_id = self.names.index(diag)
            else:
                raise KeyError(f"diagnostic {diag} could not be found in {self.names}")
        elif isinstance(diag, int):
            if diag >= 0 and diag < len(self):
                diag_id = diag
            else:
                raise IndexError(
                    f"diagnostic id was {diag} - must be within 0 and {len(self)-1}"
                )
        else:
            raise KeyError("must be str or int")
        return diag_id

    def __len__(self) -> int:
        return len(self.diagnostics)

    def __iter__(self):
        return iter(self.diagnostics.values())

