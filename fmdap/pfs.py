# from collections import namedtuple
import pandas as pd
import mikeio


class Pfs:
    def __init__(self, pfs_file=None) -> None:
        self.d = None
        self._sections = None
        self._model_errors = None
        self._measurements = None
        self._diagnostics = None

        if pfs_file:
            self.d = self._pfs2dict(pfs_file)

    @staticmethod
    def _pfs2dict(pfs_file):
        return mikeio.Pfs(pfs_file)._data

    @property
    def dda(self):
        return self.d["DATA_ASSIMILATION_MODULE"]

    @property
    def sections(self):
        if self._sections is None:
            self._sections = self._get_DA_sections()
        return self._sections

    @property
    def model_errors(self):
        if self._model_errors is None:
            self._model_errors = self._get_model_errors_df()
        return self._model_errors

    @property
    def measurements(self):
        if self._measurements is None:
            self._measurements = self._get_measurements_df()
        return self._measurements

    @property
    def diagnostics(self):
        if self._diagnostics is None:
            self._diagnostics = self._get_diagnostics_df()
        return self._diagnostics

    def _get_DA_sections(self):
        return list(self.dda.keys())

    def __getitem__(self, key):
        return self.dda.get(key.upper())

    def __getattr__(self, key):
        return self.dda.get(key.upper())

    def _get_model_errors_df(self, dda=None):
        if dda is None:
            dda = self.dda
        sec = dda.get("MODEL_ERROR_MODEL")
        if sec is None:
            raise KeyError(
                "'MODEL_ERROR_MODEL' section could not be found in dictionary!"
            )
        n_meas = int(sec.get("number_of_model_errors", 0))

        raw = {}
        for j in range(1, n_meas + 1):
            me = sec[f"MODEL_ERROR_{j}"].copy()
            ef = me.pop("Error_Formulation")
            raw[j] = {**me, **ef}
        # TODO: manually add name if not included
        # TODO: make type, perturbation_type, propagation_type etc enum
        return pd.DataFrame(raw).T

    def _get_measurements_df(self, dda=None):
        if dda is None:
            dda = self.dda
        meas_sec = dda.get("MEASUREMENTS")
        if meas_sec is None:
            raise KeyError("'MEASUREMENTS' section could not be found in dictionary!")
        n_meas = int(meas_sec.get("number_of_independent_measurements", 0))

        raw = {}
        for j in range(1, n_meas + 1):
            raw[j] = meas_sec[f"MEASUREMENT_{j}"]
        # TODO: manually add default type if not included
        # TODO: make type, include, type_time_interpolation etc enum
        return pd.DataFrame(raw).T

    def _get_diagnostics_df(self, dda=None):
        if dda is None:
            dda = self.dda
        diag_sec = dda.get("DIAGNOSTICS", {}).get("OUTPUTS")
        if diag_sec is None:
            raise KeyError(
                "DIAGNOSTICS/OUTPUTS section could not be found in dictionary!"
            )
        n_diag = int(diag_sec.get("number_of_outputs", 0))

        raw = {}
        for j in range(1, n_diag + 1):
            raw[j] = diag_sec[f"OUTPUT_{j}"]

        # TODO: make type etc enum
        return pd.DataFrame(raw).T
