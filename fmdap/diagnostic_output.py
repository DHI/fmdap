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
from enum import IntEnum
import os
import warnings
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm

from mikeio import Dfs0, eum

# TODO
# http://www.data-assimilation.net/Documents/sangomaDL6.14.pdf
# 1. Check whiteness of innovations
# 2. innovation histograms. DONE
# 3. Calc innovation statistics


class DiagnosticType(IntEnum):
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
        items = [eum.ItemInfo(eum.EUMType.Undefined)]
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
        return self.df[self._member_cols].to_numpy()

    @property
    def _member_cols(self):
        return self.df.columns

    @property
    def n_members(self):
        return len(self._member_cols)

    @property
    def shape(self):
        return self.df[self._member_cols].shape

    @property
    def is_ensemble(self):
        return self.n_members > 1

    @property
    def time(self):
        """the time vector (index as datetime)"""
        return self.df.index.get_level_values(0).to_pydatetime()

    def __init__(self, df, name=None, attrs={}, eumText=None):
        self.df = df
        self.name = name
        self.attrs = attrs
        self.eumText = eumText

    def __repr__(self):
        out = [f"<{self.__class__.__name__}> {self.name} ({self.eumText})"]
        if len(self.df) == 0:
            out.append("Empty!")
        else:
            nsteps_normal, nsteps_update = self._get_nsteps_with_type()
            nsteps = nsteps_normal + nsteps_update
            # nsteps = len(self.df.index.get_level_values(0).unique())
            if nsteps == 1:
                txt = "with" if nsteps_update == 1 else "without"
                out.append(f" Time: {self.time[0]} (1 record {txt} update)")
            else:
                steptxt = f"({nsteps} steps; {nsteps_update} with updates)"
                if isinstance(self, DiagnosticIncrements):
                    steptxt = f"({nsteps} steps)"
                out.append(f" Time: {self.time[0]} - {self.time[-1]} {steptxt}")
            if self.df.index.nlevels > 1:
                npoints = len(self.df)
                out.append(
                    f" Spatially distributed points with avg {float(npoints)/float(nsteps)} points per step"
                )
            if self.is_ensemble:
                out.append(f" Ensemble with {self.n_members} members")
            vals = self.values.ravel()
            out.append(
                f" Model: {len(vals)} values from {vals.min():.4f} to {vals.max():.4f} with mean {vals.mean():.4f}"
            )
            if "Measurement" in self.df.columns:
                df = self.df[["Mean_State", "Measurement"]].dropna()
                vals = df[["Mean_State"]].to_numpy()
                out.append(
                    f" Measurements: {len(vals)} values from {vals.min():.4f} to {vals.max():.4f} with mean {vals.mean():.4f}"
                )
                rmse = self.rmse
                bias = self.bias
                stdtxt = ""
                if self.is_ensemble:
                    stdtxt = f"ensemble_std={self.ensemble_std:.4f}"
                out.append(f" Mean skill: bias={bias:.4f}, rmse={rmse:.4f} {stdtxt}")

        return str.join("\n", out)

    def _get_nsteps_with_type(self):
        df = self.df.index.value_counts()
        if self.df.index.nlevels > 1:
            new_idx = [multiidx[0] for multiidx in df.index]
            df.index = new_idx
        nvalues_per_time = df.groupby(level=0).mean()
        return sum(nvalues_per_time == 1), sum(nvalues_per_time == 2)

    @property
    def n(self):
        return len(self.df)

    def __len__(self):
        return len(self.df)

    def __copy__(self):
        return deepcopy(self)

    def copy(self):
        return self.__copy__()

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
            plt.axvline(c="0.5")

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

    def iplot(
        self, title=None, ylim=None, **kwargs,
    ):  # pragma: no cover
        import plotly.graph_objects as go

        fig = go.Figure()

        show_members = (self.n_members > 1) or ("Mean_State" not in self.df.columns)
        if show_members:
            plot_markers = "" if "Mean_State" in self.df.columns else "+markers"
            self._iplot_add_members(fig, self.df, self.n_members, plot_markers)

        self._iplot_add_mean_state(fig, self.df)
        self._iplot_add_measurement(fig, self.df)

        if isinstance(self, (DiagnosticIncrements, DiagnosticInnovations)):
            fig.add_hline(y=0.0)

        title = title if title else self.name
        fig.update_layout(title=title, yaxis_title=self.eumText, **kwargs)
        fig.update_yaxes(range=ylim)

        fig.show()

    @staticmethod
    def _iplot_add_members(fig, df, n_members, plot_markers):  # pragma: no cover
        import plotly.graph_objects as go

        for j in range(n_members):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df.iloc[:, j],
                    name=f"State {j+1}",
                    showlegend=False,
                    mode="lines" + plot_markers,
                    line=dict(color="#999999", width=1),
                    marker=dict(size=3),
                )
            )

    @staticmethod
    def _iplot_add_mean_state(fig, df):  # pragma: no cover
        import plotly.graph_objects as go

        if "Mean_State" not in df.columns:
            return

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df.Mean_State,
                name=f"Mean_State",
                showlegend=True,
                mode="lines",
                line=dict(color="#333333"),
            )
        )

    @staticmethod
    def _iplot_add_measurement(fig, df):  # pragma: no cover
        import plotly.graph_objects as go

        if "Measurement" not in df.columns:
            return

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df.Measurement,
                name=f"Measurement",
                showlegend=True,
                mode="markers",
                line=dict(color="#EE1133"),
            )
        )

    def min(self, axis=1, **kwargs):
        return self.df[self._member_cols].min(axis=axis, **kwargs)

    def max(self, axis=1, **kwargs):
        return self.df[self._member_cols].max(axis=axis, **kwargs)

    def mean(self, axis=1, **kwargs):
        return self.df[self._member_cols].mean(axis=axis, **kwargs)

    def median(self, axis=1, **kwargs):
        return self.df[self._member_cols].median(axis=axis, **kwargs)

    def std(self, axis=1, **kwargs):
        return self.df[self._member_cols].std(axis=axis, **kwargs)

    @property
    def ensemble_std(self):
        # if self.is_ensemble:
        return self.values.std(axis=1).mean()


class DiagnosticIncrements(DiagnosticDataframe):
    def __init__(self, df: pd.DataFrame, name=None, eumText=None):
        super().__init__(df, name=name, eumText=eumText)

    def plot(self, color="0.5", marker=".", legend=None, **kwargs):
        if "title" not in kwargs:
            kwargs["title"] = self.name
        axes = self.df.plot(
            legend=legend, color=color, marker=marker, ylabel=self.eumText, **kwargs
        )
        plt.axhline()
        return axes


class DiagnosticInnovations(DiagnosticDataframe):
    def __init__(self, df: pd.DataFrame, name=None, eumText=None):
        super().__init__(df, name=name, eumText=eumText)

    def plot(self, color="0.5", marker=".", legend=None, **kwargs):
        if "title" not in kwargs:
            kwargs["title"] = self.name
        axes = self.df.plot(legend=legend, color=color, marker=marker, **kwargs)
        plt.axhline()
        return axes


class DiagnosticResults(DiagnosticDataframe):
    def __init__(
        self, df: pd.DataFrame, type: DiagnosticType, name=None, eumText=None,
    ):
        super().__init__(df, name=name, eumText=eumText)
        self.type = type
        self.is_point = True if df.index.nlevels == 1 else False
        self._innovations = None
        self._comparer = None

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
    def _member_cols(self):
        return [c for c in self.df.columns if c.startswith("State_")]

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


class _DiagnosticIndexMixin:
    """Mixin handling indexing of forecast, analysis and no-update steps"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._n_updates = None
        self._iforecast = None
        self._ianalysis = None
        self._inoupdates = None
        self._increment = None
        self._forecast_at_update = None
        self._total_forecast = None
        self._analysis = None
        self._total_result = None

    @property
    def _output_class(self):
        has_meas = "Measurement" in self.df.columns
        return MeasurementDiagnostic if has_meas else DiagnosticResults

    @property
    def increment(self):
        if self.n_updates == 0:
            warnings.warn("increment only available when data contains updates!")
            return None
        if self._increment is None:
            self._increment = self._get_increment()
        return self._increment

    @property
    def forecast_at_update(self):
        if self._forecast_at_update is None:
            self._forecast_at_update = self._get_forecast_at_update()
        if self.n_updates == 0:
            warnings.warn("No updates found! forecast_at_update empty")
            return None
        return self._forecast_at_update

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
            return None
        return self._analysis

    @property
    def result(self):
        if self._total_result is None:
            self._total_result = self._get_total_result()
        return self._total_result

    def _get_forecast_at_update(self):
        """Get a diagnostic object containing forecast values only before updates"""
        df = self.df.iloc[self.idx_forecast]
        return self._output_class(
            df,
            type=self.type,
            name=f"{self.name} forecast@update",
            eumText=self.eumText,
        )

    def _get_total_forecast(self):
        """Get a diagnostic object containing no-update and forecast values"""
        df = self.df.iloc[self.idx_forecast | self.idx_no_update]
        return self._output_class(
            df, type=self.type, name=f"{self.name} forecast", eumText=self.eumText,
        )

    def _get_analysis(self):
        """Get a diagnostic object containing analysis values"""
        df = self.df.iloc[self.idx_analysis]
        return self._output_class(
            df, type=self.type, name=f"{self.name} analysis", eumText=self.eumText,
        )

    def _get_total_result(self):
        """Get a diagnostic object containing no-update and analysis values"""
        df = self.df.iloc[self.idx_analysis | self.idx_no_update]
        return self._output_class(
            df, type=self.type, name=f"{self.name} result", eumText=self.eumText,
        )

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

        self._iforecast = df.index.duplicated(keep="last")
        self._ianalysis = df.index.duplicated(keep="first")
        self._inoupdate = (~self._iforecast) & (~self._ianalysis)
        self._n_updates = len(np.where(self._ianalysis)[0])

    def _get_mean_increments(self):
        """Determine the mean increments

        Returns:
            df_increment -- a dataframe containing the mean increments
        """
        dff = self.df[["Mean_State"]].iloc[self.idx_forecast]
        dfa = self.df[["Mean_State"]].iloc[self.idx_analysis]
        df_increment = dfa.subtract(dff)
        return df_increment


class MeasurementDiagnostic(DiagnosticResults):
    @property
    def measurement(self):
        return self.df[["Measurement"]].dropna()

    @property
    def innovation(self):
        """innovation (y-Hx) object"""
        if self._innovations is None:
            self._innovations = self._get_innovation()
        return self._innovations

    def _get_innovation(self):
        df = self.df.drop(columns="Mean_State").dropna()
        dfi = -df.iloc[:, :-1].sub(df.iloc[:, -1], axis=0)
        return DiagnosticInnovations(
            dfi, name=f"{self.name} innovation", eumText=self.eumText,
        )

    @property
    def residual(self):
        cols = self._member_cols + ["Measurement"]
        df = self.df[cols].dropna()
        return df[self._member_cols] - np.vstack(df["Measurement"])

    @property
    def bias(self):
        return np.mean(self.residual.mean(axis=1))

    @property
    def rmse(self):
        resi = self.residual.mean(axis=1).to_numpy()
        return np.sqrt(np.mean(resi ** 2))

    def skill(self, **kwargs):
        return self.comparer.skill(**kwargs)

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
            df = self.df.reset_index(["x", "y"])
            mod = fmskill.ModelResult(df[["x", "y", "Mean_State"]], type="track")
            obs = fmskill.TrackObservation(
                df[["x", "y", "Measurement"]], name=self.name
            )

        con = fmskill.Connector(obs, mod)
        cc = con.extract()
        if (cc is None) or (len(cc) == 0):
            return cc  # None
        else:
            return cc[0]


class MeasurementPointDiagnostic(_DiagnosticIndexMixin, MeasurementDiagnostic):
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

    def _get_comparer(self):
        if self.has_updates:
            cf = self.forecast.comparer
            cfau = self.forecast_at_update.comparer
            ca = self.analysis.comparer
            cr = self.result.comparer
            return cf + cfau + ca + cr
        else:
            return self.result.comparer

    def scatter(self, **kwargs):
        return self.result.scatter(**kwargs)

    def hist(self, **kwargs):
        return self.result.hist(**kwargs)

    def ecdf(self, **kwargs):
        return self.result.ecdf(**kwargs)


class MeasurementDistributedDiagnostic(MeasurementPointDiagnostic):
    def __init__(self, df, name, eumItem=None, filename=None):
        self._xy_name = list(df.columns[:2])
        super().__init__(df=df, name=name, eumItem=eumItem, filename=filename)
        self.df = self.df.set_index(["x", "y"], append=True)

    def _new_column_names(self, columns):
        n_members = len(columns) - 4
        cols = ["x", "y"]
        for j in range(n_members):
            cols.append(f"State_{j+1}")
        cols.append("Mean_State")
        cols.append("Measurement")
        return cols


class NonMeasurementPointDiagnostic(_DiagnosticIndexMixin, DiagnosticResults):
    def __init__(self, df, name, eumItem=None, filename=None):
        type = DiagnosticType.NonMeasurementPoint
        eumText = "" if eumItem is None else _get_eum_text(eumItem)
        super().__init__(df=df, type=type, name=name, eumText=eumText)
        self.filename = filename
        self.df.columns = self._new_column_names(df.columns)

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
