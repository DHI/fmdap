"""The `diagnostic_output` module contains methods and classes for 
working with MIKE FM DA diagnostic outputs.

The entrance point is always the `read_diagnostic()` method which 
will return a specialized object depending on the type of diagnostic output.

The returned object will be of type:

See also
--------
MeasurementPointDiagnostic
MeasurementDistributedDiagnostic
NonMeasurementPointDiagnostic
GlobalAssimilationDiagnostic

Examples
--------
>>> import fmdap
>>> d = fmdap.read_diagnostic("Diagnostics_Drogden_OI.dfs0", name="Drogden")
>>> d.increment.hist()
>>> d.analysis.innovation.hist()
>>> d.result.plot()
"""
from enum import Enum
import os
import warnings
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm

from mikeio import Dfs0, eum

# TODO
# http://www.data-assimilation.net/Documents/sangomaDL6.14.pdf
# 1. Check whiteness of innovations
# 2. innovation histograms. DONE
# 3. Calc innovation statistics


class DiagnosticType(Enum):
    Measurement = 1
    NonMeasurementPoint = 2
    GlobalAssimilationStatistics = 3


def read_diagnostic(filename, name=None):
    """Read diagnostic output dfs0 file"""
    if isinstance(filename, pd.DataFrame):
        df = filename  # filename is actually a DataFrame
        filename = ""
        if name is None:
            name = "Diagnostic"
        items = [eum.EumItem(eum.EUMType.Undefined)]
    else:
        dfs = Dfs0(filename)
        if name is None:
            name = os.path.basename(filename).split(".")[0]
        items = dfs.items
        df = dfs.to_dataframe()
    df.index.name = "time"

    type = _infer_diagnostic_type(df)
    if type == DiagnosticType.Measurement:
        if _is_point_subtype(df):
            return MeasurementPointDiagnostic(df, name, items[-1], filename)
        else:
            return MeasurementDistributedDiagnostic(df, name, items[-1], filename)
    elif type == DiagnosticType.NonMeasurementPoint:
        return NonMeasurementPointDiagnostic(df, name, items[-1], filename)
    elif type == DiagnosticType.GlobalAssimilationStatistics:
        return GlobalAssimilationDiagnostic(df, name, items, filename)


def _is_point_subtype(df) -> bool:
    cols = list(df.columns)
    if (cols[0][0:9].lower() == "longitude") or (cols[0][0:7].lower() == "easting"):
        return False
    else:
        return True


def _infer_diagnostic_type(df) -> DiagnosticType:
    """Determine diagnostic type based on item names

    Returns:
        diag_type -- diagnostic type (1, 2 or 3)
    """
    cols = list(df.columns)
    if cols[-1][0:10].lower() == "mean state":
        return DiagnosticType.NonMeasurementPoint
    elif (cols[-1][0:11].lower() == "measurement") & (
        cols[-2][0:10].lower() == "mean state"
    ):
        return DiagnosticType.Measurement
    elif cols[0][0:18] == "points assimilated":
        return DiagnosticType.GlobalAssimilationStatistics
    else:
        raise Exception(
            f"Diagnostic type could not be determined - based on item names: {cols}"
        )


class DiagnosticDataframe:
    @property
    def values(self):
        """all values as a nd array"""
        return self.df.to_numpy()

    @property
    def time(self):
        """the time vector (index as datetime)"""
        return self.df.index.to_pydatetime()

    def __init__(self, df, name=None, eumText=None):
        self.df = df
        self.name = name
        self.eumText = eumText

    @property
    def n(self):
        return len(self.df)

    def __len__(self):
        return len(self.df)

    def hist(self, bins=100, show_Gaussian=True, **kwargs):
        """plot histogram of values using plt.hist()

        Parameters
        ----------
        bins : int, optional
            histgram bins, by default 100
        """
        _ = plt.hist(self.values.ravel(), bins=bins, density=1, **kwargs)

        if show_Gaussian:
            xx = np.linspace(self.values.min(), self.values.max(), 300)
            yy = norm.pdf(xx, self.values.mean(), self.values.std())
            plt.gca().plot(xx, yy, "--", label="Gaussian")

        plt.xlabel(self.eumText)
        plt.title(f"Histogram of {self.name}")

    def ecdf(self, show_Gaussian=True, **kwargs):
        _, ax = plt.subplots(**kwargs)
        _ecdf = ECDF(self.values.ravel())

        if show_Gaussian:
            xx = np.linspace(self.values.min(), self.values.max(), 300)
            yy = norm.cdf(xx, self.values.mean(), self.values.std())
            ax.plot(xx, yy, "--", label="Gaussian")

        ax.plot(_ecdf.x, _ecdf.y, label=self.name)

        if show_Gaussian:
            plt.legend()
        plt.xlabel(self.eumText)
        plt.title(f"CDF of {self.name}")

    def min(self, **kwargs):
        return self.df.min(**kwargs)

    def max(self, **kwargs):
        return self.df.max(**kwargs)

    def mean(self, **kwargs):
        return self.df.mean(**kwargs)

    def median(self, **kwargs):
        return self.df.median(**kwargs)

    #        return np.median(self.values, **kwargs)

    def std(self, **kwargs):
        return self.df.std(**kwargs)


#        return np.std(self.values, **kwargs)


class DiagnosticIncrements(DiagnosticDataframe):
    def __init__(self, df: pd.DataFrame, name=None, eumText=None):
        super().__init__(df, name=name, eumText=eumText)

    def plot(self, color="0.5", marker=".", legend=None, **kwargs):
        if "title" not in kwargs:
            kwargs["title"] = self.name
        return self.df.plot(
            legend=legend, color=color, marker=marker, ylabel=self.eumText, **kwargs
        )


class DiagnosticInnovations(DiagnosticDataframe):
    def __init__(self, df: pd.DataFrame, name=None, eumText=None):
        super().__init__(df, name=name, eumText=eumText)

    def plot(self, color="0.5", marker=".", legend=None, **kwargs):
        if "title" not in kwargs:
            kwargs["title"] = self.name
        return self.df.plot(legend=legend, color=color, marker=marker, **kwargs)


class DiagnosticResults(DiagnosticDataframe):
    def __init__(
        self, df: pd.DataFrame, type: DiagnosticType, name=None, eumText=None,
    ):
        super().__init__(df, name=name, eumText=eumText)
        self.type = type
        self._n_members = None
        self.is_point = True if df.index.nlevels == 1 else False
        self._innovations = None
        self._comparer = None

    @property
    def values(self):
        return self.df[self._member_cols].to_numpy()

    @property
    def measurement(self):
        if self.has_measurement:
            return self.df[["Measurement"]].dropna()
        else:
            warnings.warn("Only MeasurementDiagnostics has measurement")
            return None

    @property
    def has_measurement(self):
        return self.type == DiagnosticType.Measurement

    @property
    def ensemble(self):
        if self.is_ensemble:
            return self.df[self._member_cols]
        else:
            return None

    @property
    def n_members(self):
        if self._n_members is None:
            self._n_members = len(self._member_cols)
        return self._n_members

    @property
    def is_ensemble(self):
        return self.n_members > 1

    @property
    def _member_cols(self):
        return [c for c in self.df.columns if c.startswith("State_")]

    @property
    def innovation(self):
        """innovation (y-Hx) object"""
        if self.type == DiagnosticType.NonMeasurementPoint:
            warnings.warn("innovations only available for MeasurementDiagnostics file")
            return None
        if self._innovations is None:
            self._innovations = self._get_innovation()
        return self._innovations

    def _get_innovation(self):
        df = self.df.drop(columns="Mean_State").dropna()
        dfi = -df.iloc[:, :-1].sub(df.iloc[:, -1], axis=0)
        return DiagnosticInnovations(
            dfi, name=f"{self.name} innovation", eumText=self.eumText,
        )

    def plot(self, figsize=(10, 5), **kwargs):
        _, ax = plt.subplots(figsize=figsize, **kwargs)

        dfe = self.df[self._member_cols]
        dfe.columns = ["_" + c for c in dfe.columns]  # to hide legend

        dfe.plot(color="0.8", ax=ax, legend=False)
        self.df[["Mean_State"]].plot(color="0.2", ax=ax)
        if self.has_measurement:
            self.measurement.plot(
                color="red", marker=".", markersize=8, linestyle="None", ax=ax,
            )
        ax.set_ylabel(self.eumText)
        ax.set_title(self.name)
        return ax

    def hist(self, bins=100, show_Gaussian=False, **kwargs):
        super().hist(bins=bins, show_Gaussian=show_Gaussian, **kwargs)

    @property
    def skill(self):
        return self.comparer.skill()

    def scatter(self, **kwargs):
        return self.comparer.scatter(**kwargs)

    @property
    def comparer(self):
        if self._comparer is None:
            self._comparer = self._get_comparer()
        return self._comparer

    def _get_comparer(self):
        import fmskill

        if self.is_point:
            mod = fmskill.ModelResult(self.df[["Mean_State"]])
            obs = fmskill.PointObservation(self.df[["Measurement"]], name=self.name)
        else:
            mod = fmskill.ModelResult(self.df[["x", "y", "Mean_State"]], type="track")
            obs = fmskill.TrackObservation(
                self.df[["x", "y", "Measurement"]], name=self.name
            )

        con = fmskill.Connector(obs, mod)

        return con.extract()[0]


class _DiagnosticIndexMixin:
    """Mixin handling indexing of forecast, analysis and no-update steps"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._n_updates = None
        self._iforecast = None
        self._ianalysis = None
        self._inoupdates = None
        self._increment = None
        self._total_forecast = None
        self._analysis = None
        self._total_result = None

    @property
    def increment(self):
        if self._increment is None:
            self._increment = self._get_increment()
        return self._increment

    @property
    def forecast(self):
        if self._total_forecast is None:
            self._total_forecast = self._get_total_forecast()
        return self._total_forecast

    @property
    def analysis(self):
        if self._analysis is None:
            self._analysis = self._get_analysis()
        if self.n_updates == 0:
            warnings.warn("No updates found! analysis empty")
        return self._analysis

    @property
    def result(self):
        if self._total_result is None:
            self._total_result = self._get_total_result()
        return self._total_result

    def _get_total_forecast(self):
        """Get a diagnostic object containing no-update and forecast values"""
        df = self.df.iloc[self.idx_forecast | self.idx_no_update]
        return DiagnosticResults(
            df, type=self.type, name=f"{self.name} forecast", eumText=self.eumText,
        )

    def _get_analysis(self):
        """Get a diagnostic object containing analysis values"""
        df = self.df.iloc[self.idx_analysis]
        return DiagnosticResults(
            df, type=self.type, name=f"{self.name} analysis", eumText=self.eumText,
        )

    def _get_total_result(self):
        """Get a diagnostic object containing no-update and analysis values"""
        df = self.df.iloc[self.idx_analysis | self.idx_no_update]
        return DiagnosticResults(
            df, type=self.type, name=f"{self.name} analysis", eumText=self.eumText,
        )

    def get_iforecast_from_ianalysis(self, ianalysis):
        nt = len(ianalysis)
        iforecast = np.zeros(nt, dtype=bool)
        iforecast[0:-1] = ianalysis[1:]
        return iforecast

    def _get_increment(self):
        """Determine all increments"""

        dff = self.df[self._member_cols].iloc[self.idx_forecast]
        dfa = self.df[self._member_cols].iloc[self.idx_analysis]
        df_increment = dfa.subtract(dff)
        return DiagnosticIncrements(
            df_increment, name=f"{self.name} increment", eumText=self.eumText,
        )

    @property
    def idx_forecast(self):
        """index before assimilation updates (forecast)"""
        if self._iforecast is None:
            self._idx_at_updates()
        return self._iforecast

    @property
    def idx_analysis(self):
        """index after assimilation updates (analysis)"""
        if self._ianalysis is None:
            self._idx_at_updates()
        return self._ianalysis

    @property
    def idx_no_update(self):
        """index when there is no assimilation updates"""
        if self._inoupdate is None:
            self._idx_at_updates()
        return self._inoupdate

    @property
    def has_updates(self):
        """has file any assimilation updates (duplicate index)"""
        return self.n_updates > 0

    @property
    def n_updates(self):
        """number of assimilation updates"""
        if self._n_updates is None:
            self._idx_at_updates()
        return self._n_updates

    def _idx_at_updates(self, df=None):
        """Find index of updates in DataFrame

        Determines:
            _iforecast -- index before updates (forecast)
            _ianalysis -- index after updates (analysis)
            _inoupdate -- index when there were no update
        """
        if df is None:
            df = self.df

        time = df.index.to_pydatetime()
        nt = len(time)
        dt = np.diff(time)
        ii = dt == datetime.timedelta(0)  # find repeated datetimes

        self._iforecast = np.zeros(nt, dtype=bool)
        self._ianalysis = np.zeros(nt, dtype=bool)

        if len(ii) == None:
            # print("No updates were found in diagnostic file")
            self._inoupdate = np.ones(nt, dtype=bool)
            self._n_updates = 0
        else:
            self._iforecast[0:-1] = ii
            self._ianalysis[1:] = ii
            self._inoupdate = (self._iforecast | self._ianalysis) != True
            self._n_updates = len(self._iforecast[self._iforecast == True])

    def _get_mean_increments(self):
        """Determine the mean increments

        Returns:
            df_increment -- a dataframe containing the mean increments
        """
        dff = self.df[["Mean_State"]].iloc[self.idx_forecast]
        dfa = self.df[["Mean_State"]].iloc[self.idx_analysis]
        df_increment = dfa.subtract(dff)
        return df_increment


class MeasurementPointDiagnostic(_DiagnosticIndexMixin, DiagnosticResults):
    def __init__(self, df, name, eumItem=None, filename=None):
        type = DiagnosticType.Measurement
        eumText = "" if eumItem is None else _get_eum_text(eumItem)
        super().__init__(df=df, type=type, name=name, eumText=eumText)
        self.filename = filename
        self.df.columns = self._new_column_names(df.columns)

    def _new_column_names(self, columns):
        n_members = len(columns) - 2
        cols = [f"State_{j+1}" for j in range(n_members)]
        cols.append("Mean_State")
        cols.append("Measurement")
        return cols

    def __repr__(self):
        out = []
        out.append(f"<{type(self).__name__}>")
        if type is not None:
            out.append(f"{self.type}")
        return str.join("\n", out)

    def _get_comparer(self):
        if self.has_updates:
            cf = self.forecast.comparer
            ca = self.analysis.comparer
            return cf + ca
        else:
            return super().comparer


class MeasurementDistributedDiagnostic(_DiagnosticIndexMixin, DiagnosticResults):
    def __init__(self, df, name, eumItem=None, filename=None):
        type = DiagnosticType.Measurement
        # self.df = df
        # self.name = name
        eumText = "" if eumItem is None else _get_eum_text(eumItem)
        super().__init__(df=df, type=type, name=name, eumText=eumText)
        self.filename = filename
        self._xy_name = list(df.columns[:2])
        self.df.columns = self._new_column_names(df.columns)
        # self._set_eum_info(eumItems[-1])
        self.df = self.df.set_index(["x", "y"], append=True)

    def _new_column_names(self, columns):
        n_members = len(columns) - 4
        cols = ["x", "y"]
        for j in range(n_members):
            cols.append(f"State_{j+1}")
        cols.append("Mean_State")
        cols.append("Measurement")
        return cols

    def _get_xy_names_and_types(self, items):
        raise NotImplementedError()

    def _get_comparer(self):
        if self.has_updates:
            cf = self.forecast.comparer
            ca = self.analysis.comparer
            return cf + ca
        else:
            return super().comparer


class NonMeasurementPointDiagnostic(DiagnosticResults, _DiagnosticIndexMixin):
    def __init__(self, df, name, eumItem=None, filename=None):
        # super().__init__(df, name, filename)
        self.type = DiagnosticType.NonMeasurementPoint
        self.df = df
        self.name = name
        self.filename = filename
        self.df.columns = self._new_column_names(df.columns)
        self.eumText = "" if eumItem is None else _get_eum_text(eumItem)

    def _new_column_names(self, columns):
        n_members = len(columns) - 1
        cols = [f"State_{j+1}" for j in range(n_members)]
        cols.append("Mean_State")
        return cols


class GlobalAssimilationDiagnostic(DiagnosticDataframe):
    def __init__(self, df, name, items, filename=None):
        self.type = DiagnosticType.GlobalAssimilationStatistics
        self.df = df
        self.name = name
        self.items = items
        self.filename = filename


def _unit_display_name(name: str) -> str:
    """Display name

    Examples
    --------
    >>> unit_display_name("meter")
    m
    """
    return name.replace("meter", "m").replace("_per_", "/").replace("sec", "s")


def _get_eum_text(eumType, eumUnit=None):
    if eumType is None:
        return ""
    if isinstance(eumType, eum.ItemInfo):
        item = eumType
        eumType = item.type
        eumUnit = item.unit
    elif not isinstance(eumType, eum.eumType):
        raise TypeError("Input must be either ItemInfo or EumType")
    txt = f"{eumType.display_name}"
    if eumType != eum.EUMType.Undefined:
        unit = eumUnit.display_name
        txt = f"{txt} [{_unit_display_name(unit)}]"
    return txt
