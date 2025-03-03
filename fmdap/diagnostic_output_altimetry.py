import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

import mikeio


class DiagnosticOutputAltimetry:
    dfd = None  # dataframe of diagnostic data
    dfo = None  # dataframe of observations
    dfda = None  # dataframe of DA steps
    dfqa = None  # dataframe of non-DA steps
    is_DA = None
    nc1 = None  # boundary nodes
    msh = None  # model mesh

    def __init__(self):
        self._dfs = None

    def read(self, file_diag, file_obs, obs_col_name="adt_dhi"):
        """Read diagnostic output dfs0 and associated observation dfs0 and store as data frames

        Arguments:
            file_diag -- path to the diagnostic dfs0 file
            file_obs -- path to the observation dfs0 file
        """

        # Load diagnostic track output from MIKE FM DA
        self.dfd = mikeio.open(file_diag).to_dataframe()
        self.dfd["ModelTime"] = self.dfd.index
        cols = self.dfd.columns.values
        for j, col in enumerate(cols):
            if col.startswith("Measurement"):
                col = "Measurement"
            if col.startswith("Mean State"):
                col = "Mean_State"
            cols[j] = col
        self.dfd.columns = cols

        # Load original observation track data
        dfo = mikeio.open(file_obs).to_dataframe()  # self._dfs.
        # limit obs data to +- 2hours of diagnostic file
        t0 = self.dfd.index[0] - datetime.timedelta(hours=2)
        t1 = self.dfd.index[-1] + datetime.timedelta(hours=2)

        # rename columns in dfo
        cols = dfo.columns.values
        if len(cols) > 3:
            cols_primary = [cols[0], cols[1], obs_col_name]
            cols_secondary = [c for c in cols if c not in cols_primary]
            cols_selected = [*cols_primary, *cols_secondary]
            dfo = dfo[cols_selected]

        cols = dfo.columns.values
        cols[0:3] = ["lon", "lat", "track_observation"]
        dfo.columns = cols
        dfo = dfo.dropna(subset=["lon", "lat"])

        self.dfo = dfo[(dfo.index > t0) & (dfo.index < t1)]

        self.dfda = None
        self.dfqa = None

    def remove_points_outside_mesh(self):
        if self.msh is None:
            raise ValueError("mesh has not been provided. Please use read_mesh() first")

        xy = np.vstack([self.dfo.iloc[:, 0].values, self.dfo.iloc[:, 1].values]).T
        inside = self.msh.geometry.contains(xy)
        self.dfo = self.dfo[inside]

    def process(self, col_no_DA="mike_wl", track_split_time=3):
        self.dfo = self.assign_track_id(self.dfo, max_jump=track_split_time)

        dfd = self.dfd
        nrows = len(dfd)
        tvec = (dfd.index - dfd.index[0]).total_seconds().values
        xvec = dfd.iloc[:, 0].values
        yvec = dfd.iloc[:, 1].values
        print(f"Will now identify DA points...")
        is_analysis = [
            self.has_duplicate_txy_before(tvec, xvec, yvec, j) for j in range(nrows)
        ]
        is_forecast_at_DA = [
            self.has_duplicate_txy_after(tvec, xvec, yvec, j) for j in range(nrows)
        ]
        is_DA_step = np.bitwise_or(is_forecast_at_DA, is_analysis)
        self.dfd["is_DA_step"] = is_DA_step
        self.dfd["is_analysis"] = is_analysis
        print(f"Identified {sum(is_analysis)} observation points used for DA")
        self.is_DA = np.any(is_DA_step)

        print(f"Will now match observation points from the two dataframes...")
        idx1 = self.get_all_obs_idx()
        self.dfd["obs_pt_idx"] = idx1
        nmatch = len(np.unique(idx1[idx1 >= 0]))
        print(f"Found {nmatch} matches")

        print(f"Will now create dfda dataframe...")
        self.create_dfda(col_no_DA=col_no_DA)

        print(f"Will now create QA dataframe dfqa...")
        self.create_dfqa(col_no_DA=col_no_DA)
        print(f"DONE")

    def read_dfda(self, filename):
        dtypes = {
            "Longitude": np.float64,
            "Latitude": np.float64,
            "mean_f": np.float64,
            "mean_a": np.float64,
            "std_f": np.float64,
            "std_a": np.float64,
            "no_DA": np.float64,
            "obs": np.float64,
            "super_obs": np.float64,
            "track_id": np.int32,
        }
        dfda = pd.read_csv(filename, parse_dates=[0, 3], index_col=0, dtype=dtypes)

        # estimate offset and apply to obs
        tmp = (
            dfda.loc[dfda.super_obs.notnull(), "super_obs"]
            - dfda.loc[dfda.super_obs.notnull(), "obs"]
        ).values.astype("float")
        offset = np.round(np.nanmedian(tmp), 7)
        print(f"Estimated offset: Median difference between obs and superobs, {offset}")
        dfda["obs"] = dfda["obs"] + offset
        self.dfda = dfda

    def create_dfda(self, col_no_DA="mike_wl"):
        dfo = self.dfo
        dfd = self.dfd
        # for each row of dfd find corresponding row (index) in dfo
        cols = [
            "Longitude",
            "Latitude",
            "model_time",
            "mean_f",
            "std_f",
            "mean_a",
            "std_a",
            "no_DA",
            "obs",
            "super_obs",
            "track_id",
        ]
        dfda = pd.DataFrame(index=dfo.index, columns=cols)

        # insert data from obs data frame
        dfda["Longitude"] = dfo.iloc[:, 0]
        dfda["Latitude"] = dfo.iloc[:, 1]
        dfda["obs"] = dfo.track_observation
        dfda["track_id"] = dfo.track_id
        if (col_no_DA is not None) and (col_no_DA in dfo.columns):
            dfda["no_DA"] = dfo[col_no_DA]

        # Prepare forecast and analysis data
        # calc std and keep only relevant cols
        is_state = lambda colname: colname.startswith("State ")
        state_cols = list(filter(is_state, dfd.columns))
        dff = dfd[np.bitwise_and(dfd.is_DA_step, ~dfd.is_analysis)].copy()
        dff["std"] = dff[state_cols].std(axis=1)
        dff = dff[["ModelTime", "Mean_State", "std", "Measurement", "obs_pt_idx"]]
        dfa = dfd[dfd.is_analysis].copy()
        dfa["std"] = dfa[state_cols].std(axis=1)
        dfa = dfa[["Mean_State", "std"]]

        # Relate pts in different data frames
        # dff/dfa will always have nearest obs_pt_idx (but non-unique)
        # NO!
        idx2 = dff.obs_pt_idx.values.astype(int)
        first_idx = [np.where(idx2 == j)[0][0] for j in np.unique(idx2[idx2 >= 0])]

        # Not all obs points will be in dff/dfa (superobs/discarded)
        # Some will be "nearest" to several dff/dfa points
        # Find for each obs point, the number of associated DA-pts
        n_DA_pts = np.bincount(idx2[idx2 >= 0], minlength=len(dfda))  # per obs pt

        # keep only first-idx data
        dff = dff.iloc[first_idx]
        dfa = dfa.iloc[first_idx]

        # set new values vectorized:
        dff = dff.drop(columns=["obs_pt_idx"])
        dfda.loc[
            n_DA_pts > 0, ["model_time", "mean_f", "std_f", "super_obs"]
        ] = dff.values
        dfda.loc[n_DA_pts > 0, ["mean_a", "std_a"]] = dfa.values

        # estimate offset and apply to obs
        tmp = (
            dfda.loc[n_DA_pts > 0, "super_obs"] - dfda.loc[n_DA_pts > 0, "obs"]
        ).values.astype("float")
        offset = np.round(np.nanmedian(tmp), 7)
        print(f"Estimated offset: Median difference between obs and superobs, {offset}")
        dfda["obs"] = dfda["obs"] + offset

        self.dfda = dfda

    def create_dfqa(self, col_no_DA="mike_wl"):
        dfo = self.dfo
        dfd = self.dfd
        # for each row of dfd find corresponding row (index) in dfo
        cols = [
            "Longitude",
            "Latitude",
            "model_time",
            "mean_f",
            "std_f",
            "no_DA",
            "obs",
            "super_obs",
            "track_id",
        ]
        dfqa = pd.DataFrame(index=dfo.index, columns=cols)

        # insert data from obs data frame
        dfqa["Longitude"] = dfo.iloc[:, 0]
        dfqa["Latitude"] = dfo.iloc[:, 1]
        dfqa["obs"] = dfo.track_observation
        dfqa["track_id"] = dfo.track_id
        if (col_no_DA is not None) and (col_no_DA in dfo.columns):
            dfqa["no_DA"] = dfo[col_no_DA]

        # Prepare forecast and analysis data
        # calc std and keep only relevant cols
        is_state = lambda colname: colname[0:6] == "State "
        state_cols = list(filter(is_state, dfd.columns))
        dff = dfd[~dfd.is_DA_step].copy()
        dff["std"] = dff[state_cols].std(axis=1)
        dff = dff[["ModelTime", "Mean_State", "std", "Measurement", "obs_pt_idx"]]

        # Relate pts in different data frames
        idx2 = dff.obs_pt_idx.values.astype(int)
        # first_idx = [np.where(idx2==j)[0][0] for j in np.unique(idx2[idx2>=0])]
        get_mid_elem = lambda a: a[int(len(a) / 2)]
        nearest_idx = [
            get_mid_elem(np.where(idx2 == j)[0]) for j in np.unique(idx2[idx2 >= 0])
        ]

        # Not all obs points will be in dff (superobs/discarded)
        # Some will be "nearest" to several dff points
        # Find for each obs point, the number of associated DA-pts
        n_QA_pts = np.bincount(idx2[idx2 >= 0], minlength=len(dfqa))  # per obs pt

        # keep only nearest data
        dff = dff.iloc[nearest_idx]

        # set new values vectorized:
        dff = dff.drop(columns=["obs_pt_idx"])
        dfqa.loc[
            n_QA_pts > 0, ["model_time", "mean_f", "std_f", "super_obs"]
        ] = dff.values

        # estimate offset and apply to obs
        tmp_diff = (
            dfqa.loc[n_QA_pts > 0, "super_obs"] - dfqa.loc[n_QA_pts > 0, "obs"]
        ).values.astype("float")
        offset = np.round(np.nanmedian(tmp_diff), 7)
        print(f"Estimated offset: Median difference between obs and superobs, {offset}")
        dfqa["obs"] = dfqa["obs"] + offset
        self.dfqa = dfqa

    @staticmethod
    def get_space_time_dist2(txy, tvec, xvec, yvec):
        # find space-time distance SQUARED to all points in vector
        # assuming speed of 7000m/s and x, y in degrees (and 1degree=1e5m)
        td = txy[0] - tvec  # relative time axis in seconds
        xd = txy[1] - xvec
        yd = txy[2] - yvec
        d2 = (7000 * td) ** 2 + (1e5 * xd) ** 2 + (1e5 * yd) ** 2
        return d2

    @staticmethod
    def get_idx_nearest_xy(txy, tvec, xvec, yvec):
        # find nearest position within +-1 hour
        td = (tvec - txy[0]).total_seconds().values  # relative time axis
        if np.min(np.abs(td)) > 3600:
            return None
        i1 = np.argmin(np.abs(td - (-3600)))  # up to 1 hour before
        i2 = np.argmin(np.abs(td - (3600)))  # up to 1 hour after

        if i2 <= i1:
            return i1

        d2 = ((txy[1] - xvec[i1 : i2 + 1])) ** 2 + ((txy[2] - yvec[i1 : i2 + 1])) ** 2
        return np.argmin(d2) + i1

    def get_all_obs_idx(self):
        # for each row of dfd find corresponding row (index) in dfo
        tvec = self.dfo.index
        xvec = self.dfo.iloc[:, 0].values
        yvec = self.dfo.iloc[:, 1].values
        # idx = np.full(len(self.dfd),pd.NA)
        idx = -np.ones(len(self.dfd), dtype=int)  # -1 indicates no pt
        j = 0
        for row in self.dfd.itertuples():
            txy = (row.Index, row[1], row[2])
            idx[j] = self.get_idx_nearest_xy(txy, tvec, xvec, yvec)
            j = j + 1
        return idx  # .astype(int)

    def has_duplicate_txy_before(self, tvec, xvec, yvec, j):
        # diagnostic track will contain duplicate rows if DA sim.
        # Seach all rows before j to see if there is a match
        if j == 0:
            return False

        txy = (tvec[j], xvec[j], yvec[j])
        d2 = self.get_space_time_dist2(txy, tvec[0:j], xvec[0:j], yvec[0:j])
        return np.min(d2) < 1e-6

    def has_duplicate_txy_after(self, tvec, xvec, yvec, j):
        # diagnostic track will contain duplicate rows if DA sim.
        # Seach all rows before j to see if there is a match

        if j >= len(tvec) - 1:
            return False

        txy = (tvec[j], xvec[j], yvec[j])
        d2 = self.get_space_time_dist2(txy, tvec[j + 1 :], xvec[j + 1 :], yvec[j + 1 :])
        return np.min(d2) < 1e-6

    def find_nearest_meas_idx(self):
        # for each row of dfd find corresponding row (index) in dfo

        self.dfd["meas_pt_id"] = pd.NA  # int(0)  # initialize

        tvec = self.dfo.index
        xvec = self.dfo.iloc[:, 0].values
        yvec = self.dfo.iloc[:, 1].values

        for i, row in self.dfd.iterrows():
            txy = (i, row.Longitude, row.Latitude)
            idx = self.get_idx_nearest_xy(txy, tvec, xvec, yvec)
            self.dfd.at[i, "meas_pt_id"] = int(idx)

        return self.dfd

    # def copy_df_to_nearest_idx(self, dfd, dfo):
    #     # for each row of dfd find corresponding row (index) in dfo
    #     dff = pd.DataFrame(index=dfo.index, columns=dfd.columns)
    #     dfa = pd.DataFrame(index=dfo.index, columns=dfd.columns)

    #     tvec = dfo.index
    #     xvec = dfo.lon.values
    #     yvec = dfo.lat.values

    #     for i, row in dfd.iterrows():
    #         txy = (i, row.Longitude, row.Latitude)
    #         idx = self.get_idx_nearest_xy(txy, tvec, xvec, yvec)
    #         if idx is not None:
    #             if row.is_analysis:
    #                 dfa.iloc[idx] = row
    #             else:
    #                 dff.iloc[idx] = row

    @staticmethod
    def assign_track_id(df, max_jump=3):
        # loop over time to find consecutive points
        # 1 step (=1second = 7.2km)

        tvec = np.asarray([dt.timestamp() for dt in df.index.to_pydatetime()])
        dtvec = np.zeros(np.size(tvec))
        nt = len(dtvec)
        dtvec[1:] = np.diff(tvec)

        ids = np.zeros(tvec.shape, dtype=int) - 1  # default is -1
        idx = 0
        ni = 0
        for j in range(nt):
            if (dtvec[j] > max_jump) & (ni > 0):
                idx = idx + 1
                ni = 0

            # only assign track id if actual data?
            # if not np.isnan(wl[j]):
            ids[j] = idx
            ni = ni + 1

        tot_tracks = idx
        df["track_id"] = ids

        print(f"Identified {tot_tracks} individual passings")
        return df

    def read_mesh(self, file_mesh):
        """Read diagnostic output dfs0 file, determine type and store as data frame

        Arguments:
            filename -- path to the dfs0 file
        """
        self.msh = mikeio.open(file_mesh)
        # self.nc1 = self.msh.get_node_coords(code=1)
        self.nc1 = self.msh.node_coordinates[self.msh.geometry.codes == 1]

    def plot_track(self, track_id):
        dfsub = self.dfda[self.dfda.track_id == track_id]

        dfsub["mean_f"].plot()
        dfsub["mean_a"].plot()
        dfsub["no_DA"].plot()
        dfsub["super_obs"].plot(linestyle="None", marker="s")
        ax1 = plt.gca()
        ax1.fill_between(
            dfsub.index,
            (dfsub.mean_f - dfsub.std_f).astype(float),
            (dfsub.mean_f + dfsub.std_f).astype(float),
            facecolor="blue",
            alpha=0.2,
        )
        ax1.fill_between(
            dfsub.index,
            (dfsub.mean_a - dfsub.std_a).astype(float),
            (dfsub.mean_a + dfsub.std_a).astype(float),
            facecolor="orange",
            alpha=0.2,
        )
        ax1.legend(["forecast", "analysis", "no_DA", "obs"])

        plt.show()

    def _plot_track_in_axis(self, dfsub, ax):
        dfsub2 = dfsub.dropna(subset=["model_time"])
        dfsub2["mean_f"].plot(ax=ax, marker=".", markersize=4)
        dfsub2["mean_a"].plot(ax=ax, marker=".", markersize=4)
        dfsub2["no_DA"].plot(ax=ax, color="gray")
        dfsub["super_obs"].plot(linestyle="None", color="red", marker="s", ax=ax)
        dfsub["obs"].plot(linestyle="None", color="darkred", marker="x", ax=ax)
        ax.fill_between(
            dfsub2.index,
            (dfsub2.mean_f - dfsub2.std_f).astype(float),
            (dfsub2.mean_f + dfsub2.std_f).astype(float),
            facecolor="blue",
            alpha=0.2,
        )
        ax.fill_between(
            dfsub2.index,
            (dfsub2.mean_a - dfsub2.std_a).astype(float),
            (dfsub2.mean_a + dfsub2.std_a).astype(float),
            facecolor="orange",
            alpha=0.2,
        )
        ax.legend(["forecast", "analysis", "no_DA", "super obs", "obs"])

    def _plot_qa_track_in_axis(self, dfsub, ax):
        dfsub2 = dfsub.dropna(subset=["model_time"])
        dfsub2["mean_f"].plot(ax=ax, marker=".", markersize=4)
        dfsub["super_obs"].plot(linestyle="None", color="red", marker="s", ax=ax)
        dfsub["obs"].plot(linestyle="None", color="darkred", marker="x", ax=ax)
        ax.fill_between(
            dfsub2.index,
            (dfsub2.mean_f - dfsub2.std_f).astype(float),
            (dfsub2.mean_f + dfsub2.std_f).astype(float),
            facecolor="blue",
            alpha=0.2,
        )
        ax.legend(["MIKE", "super obs", "obs"])

    def _plot_map_in_axis(self, dfsub, ax):
        mean_lat = 0.5 * (max(self.nc1[:, 1]) - min(self.nc1[:, 1]))
        ax.set_aspect(1.0 / np.cos(np.pi * mean_lat / 180))
        dfsub.plot.scatter(x="Longitude", y="Latitude", c="obs", colormap="jet", ax=ax)
        # ax.plot(self.nc1[:,0],self.nc1[:,1], color='gray', marker='.', markersize=2, linestyle='None')
        self.msh.plot(plot_type="outline_only", ax=ax)
        ax.plot(
            dfsub.Longitude[0],
            dfsub.Latitude[0],
            color="black",
            marker="o",
            markersize=8,
            linestyle="None",
        )

    def get_unique_model_times(self, dfsub):
        model_times = pd.to_datetime(
            np.unique(dfsub[dfsub.super_obs.notnull()].model_time.values)
        )
        return model_times

    def get_track_first_model_time(self, track_id):
        dfsub = self.dfda[self.dfda.track_id == track_id]
        model_times = self.get_unique_model_times(dfsub)
        if len(model_times) == 0:
            return None
        return model_times[0]

    def plot_track_with_map(
        self, track_id, title=None, figsize=None, show_statistics=True, unit="m"
    ):
        dfsub = self.dfda[self.dfda.track_id == track_id]
        dfsub2 = dfsub.dropna(subset=["model_time"])
        nDA = len(dfsub2)
        if nDA == 0:
            print(f"track {track_id}: no DA points to plot")
            return None

        fig, (ax1, ax2) = plt.subplots(
            1, 2, gridspec_kw={"width_ratios": [2, 1]}, figsize=figsize
        )

        # left-side: time series
        self._plot_track_in_axis(dfsub, ax1)
        if title is None:
            title = dfsub2.model_time[0].strftime("%Y-%m-%d %H:%M")
        ax1.set_title(title)

        # right-side: map
        self._plot_map_in_axis(dfsub, ax2)

        # statistics (add as title on right-side plot)
        if show_statistics:
            bias_f = (dfsub2.mean_f - dfsub2.super_obs).median()
            bias_a = (dfsub2.mean_a - dfsub2.super_obs).median()
            vals = (dfsub2.mean_f - dfsub2.super_obs).values
            rmse_f = np.sqrt(np.mean(vals**2))
            vals = (dfsub2.mean_a - dfsub2.super_obs).values
            rmse_a = np.sqrt(np.mean(vals**2))
            std_f = (dfsub2.std_f).mean()
            std_a = (dfsub2.std_a).mean()
            inno = (dfsub2.mean_a - dfsub2.mean_f).mean()
            ax2.set_title(
                f"bias_f = {bias_f:.4f}{unit}\n "
                f"bias_a = {bias_a:.4f}{unit}\n "
                f"std_f  = {std_f:.4f}{unit}\n"
                f"std_a  = {std_a:.4f}{unit}\n"
                f"rmse_f = {rmse_f:.4f}{unit}\n"
                f"rmse_a = {rmse_a:.4f}{unit}",
                loc="right",
            )

        # plt.show()
        return fig, (ax1, ax2)
