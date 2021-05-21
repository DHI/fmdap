from enum import Enum
import os
import warnings
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

from mikeio import Dfs0, eum

# TODO
# http://www.data-assimilation.net/Documents/sangomaDL6.14.pdf
# 1. Check whiteness of innovations
# 2. innovation histograms
# 3. Calc innovation statistics


class DiagnosticType(Enum):
    MeasurementPoint = 1
    NonMeasurementPoint = 2
    GlobalStatistics = 3


class DiagnosticDataframe:
    @property
    def values(self):
        """a column vector containing all values"""
        return self._values

    @property
    def time(self):
        """the time vector (index)"""
        return self.df.index.to_pydatetime()

    def __init__(self, df, name=None, eumType=None, eumUnit=None):
        self.df = df
        self.name = name
        self.eumType = eumType
        self.eumUnit = eumUnit
        self.eumText = f"{self.eumType.name} [{self.eumUnit.name}]"
        self._values = self._to_array()

    def _get_eum_text(self, eumType, eumUnit):
        if eumType is None:
            return ""
        txt = f"{eumType.display_name}"
        if eumType != eum.EUMType.Undefined:
            unit = self.itemInfo.unit.display_name
            txt = f"{txt} [{self._unit_display_name(unit)}]"
        return txt

    def _unit_display_name(self, name: str) -> str:
        """Display name

        Examples
        --------
        >>> unit_display_name("meter")
        m
        """
        return name.replace("meter", "m").replace("_per_", "/").replace("sec", "s")

    def __len__(self):
        return len(self.df)

    def _to_array(self):
        return self.df.to_numpy().reshape(-1, 1)

    def hist(self, bins=100, **kwargs):
        """plot histogram of values using plt.hist()

        Parameters
        ----------
        bins : int, optional
            histgram bins, by default 100
        """
        _ = plt.hist(self.values, bins=100, **kwargs)
        plt.xlabel(self.eumText)

    def ecdf(self, **kwargs):
        _ = plt.hist(
            self.values,
            cumulative=True,
            bins=len(self),
            density=1,
            histtype="step",
            **kwargs,
        )
        plt.xlabel(self.eumText)

    def min(self, **kwargs):
        return self.df.to_numpy().min(**kwargs)

    def max(self, **kwargs):
        return self.df.to_numpy().max(**kwargs)

    def mean(self, **kwargs):
        return self.df.to_numpy().mean(**kwargs)

    def median(self, **kwargs):
        return np.median(self.df.to_numpy(), **kwargs)

    def std(self, **kwargs):
        return np.std(self.df.to_numpy(), **kwargs)


class DiagnosticOutputIncrements(DiagnosticDataframe):
    def __init__(self, df: pd.DataFrame, name=None, eumType=None, eumUnit=None):
        super().__init__(df, name=name, eumType=eumType, eumUnit=eumUnit)

    def plot(self, color="0.5", marker=".", legend=None, **kwargs):
        return self.df.plot(
            legend=legend, color=color, marker=marker, ylabel=self.eumText, **kwargs
        )


class DiagnosticOutputInnovations(DiagnosticDataframe):
    def __init__(self, df: pd.DataFrame, name=None, eumType=None, eumUnit=None):
        super().__init__(df, name=name, eumType=eumType, eumUnit=eumUnit)

    def plot(self, color="0.5", marker=".", legend=None, **kwargs):
        return self.df.plot(legend=legend, color=color, marker=marker, **kwargs)


class DiagnosticOutputResults(DiagnosticDataframe):
    @property
    def is_ensemble(self):
        return self.n_members > 1

    @property
    def innovations(self):
        """innovation (y-Hx) object"""
        if self.type == DiagnosticType.NonMeasurementPoint:
            warnings.warn("innovations only available for MeasurementDiagnostics file")
            return None
        if self._innovations is None:
            self._innovations = self._get_innovations()
        return self._innovations

    def __init__(
        self,
        df: pd.DataFrame,
        type: DiagnosticType,
        name=None,
        eumType=None,
        eumUnit=None,
    ):
        super().__init__(df, name=name, eumType=eumType, eumUnit=eumUnit)
        self.type = type
        self._innovations = None
        ncols = len(self.df.columns)
        if type == DiagnosticType.MeasurementPoint:
            self.n_members = ncols - 2
        elif type == DiagnosticType.NonMeasurementPoint:
            self.n_members = ncols - 1
        else:
            raise ValueError("type not supported")

    def _get_innovations(self):
        if self.type == DiagnosticType.NonMeasurementPoint:
            return None
        df = self.df.drop(columns="Mean_State").dropna()
        dfi = -df.iloc[:, :-1].sub(df.iloc[:, -1], axis=0)
        return DiagnosticOutputInnovations(
            dfi, eumType=self.eumType, eumUnit=self.eumUnit
        )

    def plot(self, figsize=(10, 5), **kwargs):
        _, ax = plt.subplots(figsize=figsize, **kwargs)

        cols = [c for c in self.df.columns if c.startswith("State_")]
        dfe = self.df[cols]
        dfe.columns = ["_" + c for c in dfe.columns]  # to hide legend

        dfe.plot(color="0.8", ax=ax, legend=False)
        self.df[["Mean_State"]].plot(color="0.2", ax=ax)
        if self.type == DiagnosticType.MeasurementPoint:
            self.df[["Measurement"]].plot(
                color="red",
                marker=".",
                markersize=8,
                linestyle="None",
                ax=ax,
            )
        ax.set_ylabel(self.eumText)
        return ax


class DiagnosticOutput:
    df = None
    time = None
    n_members = 0
    # title = None
    # variable = None
    type = None
    n_updates = 0
    # _dfs = None
    # iforecast = np.zeros(0, dtype=bool)
    # ianalysis = np.zeros(0, dtype=bool)
    # inoupdate = np.zeros(0, dtype=bool)

    @property
    def is_ensemble(self):
        return self.n_members > 1

    @property
    def increments(self):
        if self.type == DiagnosticType.GlobalStatistics:
            warnings.warn("increments not available for GlobalStatistics file")
            return None
        if self._increments is None:
            self._increments = self._get_increments()
        return self._increments

    @property
    def forecast(self):
        if self.type == DiagnosticType.GlobalStatistics:
            warnings.warn("forecast not available for GlobalStatistics file")
            return None
        if self._total_forecast is None:
            self._total_forecast = self._get_total_forecast()
        return self._total_forecast

    @property
    def analysis(self):
        if self.type == DiagnosticType.GlobalStatistics:
            warnings.warn("analysis not available for GlobalStatistics file")
            return None
        if self._total_analysis is None:
            self._total_analysis = self._get_total_analysis()
        return self._total_analysis

    def __init__(self, filename=None, name=None):
        if filename is not None:
            self.read(filename)
            if name is None:
                name = os.path.basename(filename).split(".")[0]
        self.name = name
        self._increments = None
        self._total_forecast = None
        self._total_analysis = None

    def __repr__(self):
        out = []
        out.append(f"<{type(self).__name__}>")
        if type is not None:
            out.append(f"{self.type}")
        return str.join("\n", out)

    def read(self, filename):
        """Read diagnostic output dfs0 file, determine type and store as data frame

        Arguments:
            filename -- path to the dfs0 file
        """
        dfs = Dfs0(filename)
        self.df = dfs.to_dataframe()
        self.time = self.df.index.to_pydatetime()
        self.type = self._infer_diag_type()
        self._extract_info(self.df, self.type, dfs.items)

    def _extract_info(self, df: pd.DataFrame, type: DiagnosticType, items):
        if type == DiagnosticType.GlobalStatistics:
            pass
        else:
            ncols = len(df.columns)
            if type == DiagnosticType.MeasurementPoint:
                self.n_members = ncols - 2
            elif type == DiagnosticType.NonMeasurementPoint:
                self.n_members = ncols - 1

            cols = [f"State_{j+1}" for j in range(self.n_members)]
            cols.append("Mean_State")
            if type == DiagnosticType.MeasurementPoint:
                cols.append("Measurement")
            self.df.columns = cols

            self.eumType = items[0].type
            self.eumUnit = items[0].unit

    def _get_total_forecast(self):
        """Get a data frame containing no-update and forecast values"""
        iforecast, _, inoupdate = self.idx_at_updates()
        df = self.df.iloc[iforecast | inoupdate]
        return DiagnosticOutputResults(
            df,
            type=self.type,
            name=self.name,
            eumType=self.eumType,
            eumUnit=self.eumUnit,
        )

    def _get_total_analysis(self):
        """Get a data frame containing no-update and analysis values"""
        _, ianalysis, inoupdate = self.idx_at_updates()
        df = self.df.iloc[ianalysis | inoupdate]
        return DiagnosticOutputResults(
            df,
            type=self.type,
            name=self.name,
            eumType=self.eumType,
            eumUnit=self.eumUnit,
        )

    def _infer_diag_type(self, df=None) -> DiagnosticType:
        """Determine diagnostic type based on item names

        Keyword Arguments:
            df -- data frame  (default: self)

        Raises:
            Exception: if None of the three diagnostic types could be identified

        Returns:
            diag_type -- diagnostic type (1, 2 or 3)
        """
        if df is None:
            df = self.df

        cols = list(df.columns)
        if cols[-1][0:10].lower() == "mean state":
            diag_type = 2
        elif (cols[-1][0:11].lower() == "measurement") & (
            cols[-2][0:10].lower() == "mean state"
        ):
            diag_type = 1
        elif cols[0][0:18] == "points assimilated":
            diag_type = 3
        else:
            raise Exception(
                f"Diagnostic type could not be determined - based on item names: {cols}"
            )

        return DiagnosticType(diag_type)

    def idx_at_updates(self, df=None):
        """Find index of updates in data frame

        Returns:
            iforecast -- index before updates (forecast)
            ianalysis -- index after updates (analysis)
            inoupdate -- index when there were no update
        """
        if df is None:
            df = self.df

        time = df.index.to_pydatetime()
        nt = len(time)
        dt = np.diff(time)
        ii = dt == datetime.timedelta(0)  # find repeated datetimes

        if len(ii) == None:
            print("No updates were found in diagnostic file")
            return None, None, None

        iforecast = np.zeros(nt, dtype=bool)
        iforecast[0:-1] = ii
        ianalysis = np.zeros(nt, dtype=bool)
        ianalysis[1:] = ii
        inoupdate = (iforecast | ianalysis) != True
        self.n_updates = len(iforecast[iforecast == True])
        return iforecast, ianalysis, inoupdate

    def get_iforecast_from_ianalysis(self, ianalysis):
        nt = len(ianalysis)
        iforecast = np.zeros(nt, dtype=bool)
        iforecast[0:-1] = ianalysis[1:]
        return iforecast

    def _get_increments(self):
        """Determine all increments

        Returns:
            df_increment -- a dataframe containing all increments
        """
        iforecast, ianalysis, _ = self.idx_at_updates()

        state_items = [i for i in list(self.df.columns) if i.startswith("State_")]

        dff = self.df[state_items].iloc[iforecast]
        dfa = self.df[state_items].iloc[ianalysis]
        df_increment = dfa.subtract(dff)
        return DiagnosticOutputIncrements(
            df_increment, name=self.name, eumType=self.eumType, eumUnit=self.eumUnit
        )

    # def get_all_increments_as_array(self):
    #     """Determine the all increments and return as array

    #     Returns:
    #         increments -- a column vector containing all increments
    #     """
    #     df_increments = self._get_increments()
    #     return df_increments.values.reshape(-1, 1)

    def get_mean_increments(self):
        """Determine the mean increments

        Returns:
            df_increment -- a dataframe containing the mean increments
        """
        iforecast, ianalysis, _ = self.idx_at_updates()
        dff = self.df[["Mean_State"]].iloc[iforecast]
        dfa = self.df[["Mean_State"]].iloc[ianalysis]
        df_increment = dfa.subtract(dff)
        return df_increment
