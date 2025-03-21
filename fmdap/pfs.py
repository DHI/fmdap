from pathlib import Path
import numpy as np
import pandas as pd
import mikeio


class Pfs:
    def __init__(self, pfs_file: str | Path) -> None:
        self.d = None
        self._sections = None
        self._model_errors = None
        self._measurements = None
        self._diagnostics = None

        if pfs_file:
            pfs = mikeio.PfsDocument(pfs_file)
            self.d = pfs.targets[0].to_dict()

    @property
    def dda(self):
        """Dictionary of settings in DATA_ASSIMILATION_MODULE"""
        return self.d["DATA_ASSIMILATION_MODULE"]

    @property
    def sections(self):
        """List of the DA Pfs sections"""
        if self._sections is None:
            self._sections = self._get_DA_sections()
        return self._sections

    @property
    def model_errors(self):
        """DataFrame with model errors"""
        if self._model_errors is None:
            self._model_errors = self._get_model_errors_df()
        return self._model_errors

    @property
    def measurements(self):
        """DataFrame with measurements"""
        if self._measurements is None:
            self._measurements = self._get_measurements_df()
        return self._measurements

    @property
    def measurement_positions(self):
        """DataFrame with measurement positions"""
        df = self.measurements.copy()
        list_norm = [row if len(row) == 3 else row + [0] for row in df.position.to_list()]
        df[["x", "y", "z"]] = list_norm
        return df[["name", "x", "y", "z"]] 

    @classmethod
    def validate_positions(cls, mesh, df):
        """Determine if positions are inside mesh and find nearest cell centers"""
        # TODO: handle empty positions
        assert isinstance(mesh, (mikeio.Mesh, mikeio.Dfsu2DH))

        if ("x" in df) and ("y" in df):
            xy = df[["x", "y"]].to_numpy()
        elif "position" in df:
            n = len(df)
            xy = np.concatenate(df.position.to_numpy()).reshape(n, 2)
        else:
            raise ValueError(
                "Could not find 'x', 'y' or 'position' columns in DataFrame"
            )

        inside = mesh.geometry.contains(xy)
        elemid, dist = mesh.geometry.find_nearest_elements(xy, return_distances=True)
        new_positions = mesh.geometry.element_coordinates[elemid, :2]

        df2 = pd.DataFrame(index=df.index)
        if "name" in df:
            df2["name"] = df.name
        df2[["x_old", "y_old"]] = xy
        df2["inside"] = inside
        df2["dist"] = dist
        df2["elem_id"] = elemid
        df2[["x_cc", "y_cc"]] = new_positions
        return df2

    @property
    def diagnostics(self):
        """DataFrame with diagnostic outputs"""
        if self._diagnostics is None:
            self._diagnostics = self._get_diagnostics_df()
        return self._diagnostics

    def _get_DA_sections(self):
        return list(self.dda.keys())

    def __getitem__(self, key):
        return self.dda.get(key.upper())

    def __getattr__(self, key):
        return self[key]

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
