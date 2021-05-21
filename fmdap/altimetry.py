from datetime import datetime, timedelta
import time
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from mikeio import eum


class APIAuthenticationFailed(Exception):
    pass

class DownloadDataFromAPIFailed(Exception):
    pass

class InvalidSatelliteName(Exception):
    pass


class AltimetryDHI:
    API_URL = "https://altimetry-api.dhigroup.com/"
    API_user = ""
    API_password = ""
    API_token = None    
    area = None
    start_date = None
    end_date = None

    api_key = None
    df = None

    def __init__(self, user="", password=""):
        super().__init__()
        self.API_user = user
        self.API_password = password
        if (user!="") & (password!=""):
            self.API_token = self.get_access_token(user, password)

    def get_access_token(self, user="", password=""):
        if user is "":
            user = self.API_user
        if password is "":
            password = self.API_password
        if (user=="") | (password==""):
            raise Exception("username and/or password has not been provided!")

        API_CREDENTIALS = (user, password)
        r = requests.get(
            (self.API_URL + "/get-token"), 
            auth=API_CREDENTIALS
        )
        r.raise_for_status()
        token_data = r.json()
        self.API_token = token_data
        return token_data

    def check_connection(self):
        """Check connection (and credentials) to altimetry api
        
        Returns
        -------
        bool
            True if connection is ok, otherwise False
        """
        url_query = self.create_altimetry_query(
            area="lon=10.9&lat=55.9&radius=10000",
            start_date="20180115",
            end_date="20180201",
            satellites=["3a"],
        )
        try:
            df = self.get_altimetry_from_api(url_query, self.api_key)
        except:
            return False

        return len(df) > 0

    def save_to_dfs0(self, filename, df=None, satellite=None):
        """[summary]
        
        Parameters
        ----------
        filename : [type]
            [description]
        df : [type], optional
            [description], by default None
        satellite : [type], optional
            [description], by default None
        
        Raises
        ------
        Exception
            [description]
        """
        if df is None:
            df = self._data
        if satellite is not None:
            df = df[df.satellite == satellite]

        if len(df) < 1:
            raise Exception("No data in data frame")

        cols = ["lon", "lat", "adt_dhi", "swh", "wind_speed"]
        items = []
        items.append(eum.ItemInfo("Longitude", eum.EUMType.Latitude_longitude))
        items.append(eum.ItemInfo("Latitude", eum.EUMType.Latitude_longitude))
        items.append(eum.ItemInfo("Water Level", eum.EUMType.Water_Level))
        items.append(
            eum.ItemInfo("Significant Wave Height", eum.EUMType.Significant_wave_height)
        )
        items.append(eum.ItemInfo("Wind Speed", eum.EUMType.Wind_speed))

        df[cols].to_dfs0(filename, items=items)

    def get_data(
        self,
        area,
        start_date="20180101",
        end_date=datetime.now(),
        satellites="",
        quality_filter="",
    ):
        """Main function that creates url and retrieves altimetry data from api
        
        Parameters
        ----------
        area : str
            String specifying location of desired data.  The three forms allowed by the API are:
                - polygon=6.811,54.993,8.009,54.993,8.009,57.154,6.811,57.154,6.811,54.993
                - bbox=115,28,150,52
                - lon=10.9&lat=55.9&radius=100000
            A few named domains can also be used:
                - GS_NorthSea, GS_BalticSea, GS_SouthChinaSea    
        start_date : str, datetime, optional
            Start of data to be retrieved, by default '20180101'
        end_date : str, datetime, optional
            End of data to be retrieved, by default datetime.now()
        satellites : str, list of str, optional
            Satellites to be downloaded, e.g. '', '3a', 'j3, by default ''
        quality_filter : str, optional
            Name of quality filter, e.g. 'dhi_combined', by default '' meaning no filter
        
        Returns
        -------
        DataFrame
            With columns 'lon', 'lat', 'adt', 'adt_dhi', 'swh', 'swh_rms' 'wind_speed', ...
        """
        url_query = self.create_altimetry_query(
            area=area,
            start_date=start_date,
            end_date=end_date,
            quality_filter=quality_filter,
            satellites=satellites,
        )
        df = self.get_altimetry_from_api(url_query, self.api_key)
        self.area = area
        self.start_date = self.parse_date(start_date)
        self.end_date = self.parse_date(end_date)
        self._data = df
        return df

    def read_csv(self, filename):
        """read altimetry data from csv file instead of api
        
        Parameters
        ----------
        filename : str
            path to csv file
        
        Returns
        -------
        DataFrame
            With datetime index containing the altimetry data
        """
        df = pd.read_csv(filename, parse_dates=True, index_col="date")
        print(f"Succesfully read {len(df)} rows from file {filename}")

        self.start_date = df.index.min().date()
        self.end_date = df.index.max().date() + timedelta(days=1)
        self._data = df
        return df

    def get_satellites(self, df=None):
        if df is None:
            df = self._data
        return list(df.satellite.unique())

    def get_dataframe_per_satellite(self, df=None):
        if df is None:
            df = self._data
        res = {}
        sats = self.get_satellites(df)
        for sat in sats:
            dfsub = df[df.satellite == sat]
            res[sat] = dfsub  # .drop(['satellite'], axis=1)
        return res

    def assign_track_id(self, data=None, max_jump = 3):
        # loop over missions, then time to find consecutive points
        if data is None:
            data = self._data
        id = 0
        sats = self.get_satellites(data)

        # 1 step (=1second = 7.2km)
        #max_jump = 10  # what is an acceptable skip length 60?

        df = data.copy()

        if "track_id" not in df.columns:
            ids = np.zeros((len(df),), dtype=int)
            df.insert(len(df.columns), "track_id", ids, True)

        # find tracks for each satellite
        tot_tracks = 0
        for sat in sats:
            dfsub = df[df.satellite == sat]
            tvec = np.asarray([dt.timestamp() for dt in dfsub.index.to_pydatetime()])
            wl = dfsub.adt_dhi
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

            tot_tracks = tot_tracks + idx
            df.loc[df.satellite == sat, "track_id"] = ids

        print(f"Identified {tot_tracks} individual passings")

        return df


    def OLDapply_MAD_filter_swh(self, dfin, col_name='swh', flag_col_name='flag_swh'):
        # 1. check that df has track_id
        # 2. add flag_swh as col and 
 
        df = dfin.copy()

        if 'flag_swh' not in df:
            df['flag_swh'] = 0
        
        csize = 25
        minchunk = 12
        minvals = 10
        step = 10
        flag_not_ok = 4

        # get list of tracks
        track_ids = np.unique(df.track_id)
        
        # loop over all tracks
        tot_chunks = 0
        for id in track_ids: 
            dft = df[(df.track_id==id)]
            nt  = len(dft)

            # loop over chunks 
            n1 = 0
            n2 = min(nt, n1 + csize)
            local_chunks = 0
            local_idx = []
            while (n2<=nt) and ((n2-n1)>=minchunk):
                tot_chunks = tot_chunks + 1
                local_chunks = local_chunks + 1

                # define chunk and do checks
                chunk = dft.iloc[n1:n2,col_name].values
                nnotnan = np.sum(~np.isnan(chunk))
                if(nnotnan < minvals):
                    print(f'too few values: {nnotnan}; nans: {(n2-n1)-nnotnan}')
                    n1 = n1 + step 
                    n2 = n2 + step 
                    continue

                # apply filter
                not_ok = self.MAD_filter_chunk(chunk)
                
                # get (global) index of bad values
                bad_idx = np.where(not_ok) #np.where(chunkflag==4)
                if bad_idx is not None:
                    for ii in bad_idx[0]: #[0]:
                        local_idx.append(ii+n1)

                    idx = dft.iloc[bad_idx[0]+n1].index   #[0]
                    df.at[idx,'flag_swh'] = flag_not_ok

                nbad = len(bad_idx[0]) #np.count_nonzero(chunkflag)

                # debug 
                if(nbad>3):
                    print(np.round(chunk,2))
                    print(bad_idx[0])

                n0 = len(chunk) - nbad
                if nbad>1:
                    print(f'Track {id}, chunk {local_chunks}, n1={n1}, n2={n2}: n={n2-n1}, good={n0}, bad={nbad}')
                
                # loop flagged points, to check sub blocks on both sides:
                for ii in bad_idx[0]:
                    # left side of pt
                    min_subblock = 8
                    if ii > min_subblock:
                        chunkL = dft.iloc[n1:ii+n1].swh.values
                        notok = self.MAD_filter_chunk(chunkL)
                        chunkL[notok] = np.nan
                        R = np.nanstd(chunkL)/np.nanmean(chunkL)
                        if (R>0.5): 
                            # discard whole sub-block
                            notok = np.where

                    # right side of pt
                    if (n2-(ii+n1)) > min_subblock:
                        chunkR = dft.iloc[ii+n1:n2].swh.values
                        notok = self.MAD_filter_chunk_2(chunkR)


                # next chunk
                n1 = n1 + step 
                n2 = min(nt, n2 + step)
            
            # TODO: step 5 in Y&R algorithm 
            local_idx = np.unique(local_idx)
            if (local_idx is not None) and len(local_idx)>0:
                local_idx = dft.iloc[local_idx].index
                #print(local_idx)


        return df


    # @staticmethod
    # def MAD_filter_chunk(chunk, threshold=3):
    #     # apply MAD filter on chunk
    #     #n = len(chunk)
    #     flag = np.zeros(chunk.shape)
    #     md  = np.nanmedian(chunk)
    #     mad = stats.median_absolute_deviation(chunk, nan_policy='omit')
    #     notok = abs((chunk-md)/mad) >= threshold 
    #     flag[notok] = 4
    #     return flag


    def apply_MAD_filter(self, dfin, col_name='swh', flag_col_name='flag_swh', MAD_threshold=3, min_mad=0):
        # 1. check that df has track_id
        # 2. add flag_swh as col and 
 
        df = dfin.copy()

        if flag_col_name not in df:
            df[flag_col_name] = 0
        
        csize = 25
        minchunk = 20
        minvals = 8
        step = 5
        flag_not_ok = 4
        ignore_overlap = 4 #np.round((csize-step)/4)

        # get list of tracks
        track_ids = np.unique(df.track_id)
        
        # loop over all tracks        
        for id in track_ids: 
            dft = df[(df.track_id==id)]
            nt  = len(dft)

            # loop over chunks 
            n1 = 0
            n2 = min(nt, n1 + csize)
            local_chunks = 0
            local_idx = []
            while (n2<=nt) and ((n2-n1)>=minchunk):
                local_chunks = local_chunks + 1
                
                # define chunk and do checks
                chunk = dft.iloc[n1:n2][col_name].values
                nchunk = len(chunk)
                
                nnotnan = np.sum(~np.isnan(chunk))
                if(nnotnan < minvals):
                    #print(f'too few values: {nnotnan}; nans: {(n2-n1)-nnotnan}')
                    n1 = n1 + step 
                    n2 = n2 + step 
                    continue

                # process this chunk
                bad_idx = self.MAD_process_chunk(chunk, MAD_threshold, min_mad)
                nbad = len(bad_idx)

                # ignore "bad" idx in overlap region
                if nbad > 0:
                    if n1 > 0:
                        # for all chunks except first chunk: ignore start
                        #print(f'check start; before {n2-n1}: bad_idx={bad_idx}')
                        bad_idx = bad_idx[bad_idx>ignore_overlap]
                        #print(f'check start; after {n2-n1}: bad_idx={bad_idx}')
                    if n2 < nt:
                        # for all chunks except last chunk: ignore end
                        #print(f'check end; before {n2-n1}: bad_idx={bad_idx}')
                        bad_idx = bad_idx[bad_idx<(nchunk-ignore_overlap)]
                        #print(f'check end; after {n2-n1}: bad_idx={bad_idx}')

                nbad = len(bad_idx)

                # get (global) index of bad values
                if nbad > 0:
                    idx = dft.iloc[bad_idx + n1].index   
                    df.at[idx,flag_col_name] = flag_not_ok

                # debug 
                #if(nbad>1):
                #    print(np.round(chunk,2))
                #    print(bad_idx)

                n0 = len(chunk) - nbad
                if nbad>3:
                    print(f'Track {id}, chunk {local_chunks}, n1={n1}, n2={n2}: n={n2-n1}, good={n0}, bad={nbad}')
                

                # next chunk
                n1 = n1 + step 
                n2 = min(nt, n2 + step)
            
            # TODO: step 5 in Y&R algorithm 
            local_idx = np.unique(local_idx)
            if (local_idx is not None) and len(local_idx)>0:
                local_idx = dft.iloc[local_idx].index
                #print(local_idx)


        return df

    @staticmethod
    def MAD_filter_chunk(chunk, threshold=3, min_mad=0.0):
        # apply MAD filter on chunk
        md  = np.nanmedian(chunk)
        mad = stats.median_absolute_deviation(chunk, nan_policy='omit')
        mad = max(min_mad, mad)
        notok = abs((chunk-md)/mad) >= threshold
        return notok


    @staticmethod
    def MAD_process_chunk(chunk, MAD_threshold = 3, min_mad = 0, min_subblock = 6):
        nchunk = len(chunk)        
        local_idx = []

        # apply filter on chunk    
        not_ok = AltimetryDHI.MAD_filter_chunk(chunk, threshold=MAD_threshold, min_mad=min_mad)
 
        # loop flagged points, to check sub blocks on both sides:
        for ibad in np.where(not_ok)[0]:
            local_idx.append(ibad) # get index of bad values

            # left side of bad pt            
            if ibad > min_subblock:
                chunkL = chunk[0:ibad]
                not_ok = AltimetryDHI.MAD_filter_chunk(chunkL, threshold=MAD_threshold, min_mad=min_mad)
                for ibadL in np.where(not_ok)[0]:
                    #print(f'sublock-L outlier: {np.round(chunkL,2)}; ibadL={ibadL}')
                    local_idx.append(ibadL)

                # R = np.nanstd(chunkL[~not_ok])/np.nanmean(chunkL[~not_ok])
                # if (R>0.5): 
                #     print(f'sublock-L discard: {np.round(chunkL,2)}; R={R}')
                #     # discard whole sub-block
                #     for ibad in range(0,ibad):
                #         local_idx.append(ibad)

            # right side of bad pt
            if (nchunk-ibad) > min_subblock:
                chunkR = chunk[ibad:nchunk]
                not_ok = AltimetryDHI.MAD_filter_chunk(chunkR, threshold=MAD_threshold, min_mad=min_mad)
                for ibadR in np.where(not_ok)[0]:
                    #print(f'sublock-R outlier: {np.round(chunkR,2)}; ibadR={ibadR}')
                    local_idx.append(ibad+ibadR)

                # R = np.nanstd(chunkR[~not_ok])/np.nanmean(chunkR[~not_ok])
                # if (R>0.5): 
                #     print(f'sublock-R discard: {np.round(chunkR,2)}; R={R}')
                #     # discard whole sub-block
                #     for ibadR in range(ibad,nchunk):
                #         local_idx.append(ibadR)

        local_idx = np.unique(local_idx)
        return local_idx


    def clean_results(self, df=None, min_distance_land=5000, max_adt=5):
        if df is None:
            df = self._data
        nrows = len(df.index)

        df2 = df.copy()  # don't manipulate original data

        # filter ADT (=water level)
        max_range_rms = 0.5 # 0.2
        wl = ["adt", "adt_dhi"]
        # df2.loc[pd.isnull(df2.adt), 'adt_dhi'] = np.nan
        df2.loc[(df2.range_rms > max_range_rms), wl] = np.nan
        df2.loc[(abs(df2.adt) > max_adt), wl] = np.nan    # adt_dhi

        # filter wind and swh
        max_swh_rms = 2.5 #1
        max_swh = 20
        max_wind = 28
        ww = ["swh", "wind_speed"]
        df2.loc[(df2.swh_rms > max_swh_rms), ww] = np.nan
        df2.loc[(df2.swh > max_swh), ww] = np.nan
        df2.loc[(df2.wind_speed > max_wind), ww] = np.nan
        # combined: if std is larger than 1/5 of value then set to nan
        df2.loc[((df2.swh < (5 * df2.swh_rms)) & (df2.swh > 2)), ww] = np.nan

        return df2[df2.distance_from_land > min_distance_land]


    # def remove_outliers_per_passing(self, data):

    #     for tid in sats:
    #         dfsub = df[df.satellite == sat]

    # def filter_results(self, df=None, min_distance_land=5000):
    #     if df is None:
    #         df = self._data

    def print_records_per_satellite(self, df, details=1):
        if df is None:
            df = self._data
        sats = self.get_satellites()
        print(f"For the selected area between {self.start_date} and {self.end_date}:")
        for sat in sats:
            dfsub = df[df.satellite == sat]
            print(f"Satellite {sat} has {len(dfsub)} records")
            if details > 1:
                print(dfsub.drop(["lon", "lat"], axis=1).describe())

    def plot_map(self, df=None, fig_size=(12, 10)):
        """plot map of altimetry data
        
        Keyword Arguments:
            df {dataframe} -- altimetry data (default: {None})
            fig_size {tuple} -- size of figure (default: {(12,10)})
        """
        if df is None:
            df = self._data

        plt.style.use("seaborn-whitegrid")
        plt.figure(figsize=fig_size)
        markers = ["o", "x", ".", ",", "+", "v", "^", "<", ">", "s", "d"]
        j = 0
        for sat in self.get_satellites():
            dfsub = df[df.satellite == sat]
            plt.plot(dfsub.lon, dfsub.lat, markers[j], label=sat, markersize=1)
            j = j + 1
        plt.legend(numpoints=1)
        plt.title(f"Altimetry data between {self.start_date} and {self.end_date}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

    # def plot_basemap(self, fig_size=(12,10)):
    #     fig = plt.figure(figsize=fig_size)
    #     ax = fig.add_subplot(111)

    #     # , epsg=5520
    #     bmap = Basemap(llcrnrlon=6,llcrnrlat=54,urcrnrlon=30,urcrnrlat=68, resolution = 'l')
    #     #map.etopo()
    #     #map.bluemarble()
    #     bmap.fillcontinents(color='lightgray') #,lake_color='aqua')
    #     bmap.drawcoastlines()
    #     bmap.drawstates()
    #     bmap.drawcountries()
    #     bmap.drawparallels(np.arange(48,60,2.),labels=[True,False,False,False])
    #     bmap.drawmeridians(np.arange(-2,12,2.),labels=[True,False,False,True])

    #     j = 0
    #     colorlist = ['aqua','magenta','blue','green','yellow']
    #     for sat in self.get_satellites():
    #         dfsub = self._data[self._data.satellite==sat]
    #         xx, yy = bmap(dfsub.lon, dfsub.lat)
    #         bmap.plot(xx, yy, marker='.', color=colorlist[j], linewidth=0, label=sat)
    #         #plt.plot(dfsub.lon, dfsub.lat, markers[j], label=sat)
    #         j = j + 1
    #     plt.legend(numpoints=1)
    #     plt.title(f'Altimetry data between {self.start_date} and {self.end_date}')

    def create_altimetry_query(
        self,
        area="bbox=-11.913345,48.592117,12.411167,63.084148",
        satellites="sentinels",
        start_date="",
        end_date="",
        nan_value="",
        quality_filter="dhi_combined",
        numeric=False,
    ):
        """
        Create a query for satellite data as a URL pointing to the location of a CSV file with the data.

        Parameters
        ----------
        area : str
            String specifying location of desired data.  The three forms allowed by the API are:
                - polygon=6.811,54.993,8.009,54.993,8.009,57.154,6.811,57.154,6.811,54.993
                - bbox=115,28,150,52
                - lon=10.9&lat=55.9&radius=100000
            A few named domains can also be used:
                - GS_NorthSea, GS_BalticSea, GS_SouthChinaSea    
        satellites : Union[List[str], str], optional
            List of short or long names of satellite to include, an empty string, or the string 'sentinels' to specify
                the two sentinel satellites 3a and 3b. Default: 'sentinels'.
        start_date : str or datetime, optional
            First date for which data is wanted, in the format '20100101' or as an empty string. If an empty string is
                given, data starting from when it was first available is returned. Default: ''.
        end_date : str or datetime, optional
            Last date for which data is wanted, in the format '20100101' or as an empty string. If an empty string is
                given, data until the last time available is returned. Default: ''.
        nan_value : str, optional
            Value to use to indicate bad or missing data, or an empty string to use the default (-9999). Default: ''.
        quality_filter : str, optional
            Type of filter to apply before returning data. Currently, only the value 'dhi_combined' is allowed. If the
                empty string is given, no filter is applied. Default: 'dhi_combined'.
        numeric : bool, optional
            If True, return columns as numeric and return fewer columns in order to comply with the Met-Ocean on Demand 
                analysis systems. If False, all columns are returned, and string types are preserved as such. 
                Default: False.

        Returns
        -------
        str
            URL pointing to the location of data in a CSV file.
        """

        area = self.validate_area(area)
        satellite_string = self.parse_satellite_list(satellites)

        query_url = self.API_URL + "query-csv?"

        if numeric:
            numeric_string = "numeric=true"
        else:
            numeric_string = "numeric=false"

        quality_filter = self.validate_quality_filter(quality_filter)
        if quality_filter is not "":
            quality_filter = "qual_filters=" + quality_filter

        if end_date is not "":
            end_date = self.parse_date(end_date)
            end_date = "end_date=" + end_date

        if start_date is not "":
            start_date = self.parse_date(start_date)
            start_date = "start_date=" + start_date

        argument_strings = [
            arg_string
            for arg_string in [
                area,
                satellite_string,
                start_date,
                end_date,
                nan_value,
                quality_filter,
                numeric_string,
            ]
            if arg_string is not ""
        ]

        query_url = "".join([query_url, "&".join(argument_strings)])
        return query_url

    def validate_area(self, area):
        # polygon=6.811,54.993,8.009,54.993,8.009,57.154,6.811,57.154,6.811,54.993
        # bbox=115,28,150,52
        # lon=10.9&lat=55.9&radius=100000
        parsed = True
        if area == "GS_NorthSea":
            area = "polygon=9.74351,55.87496,13.17873,56.09956,13.44641,56.96066,10.72500,60.56592,9.02970,63.01088,-12.29205,63.2,-12.64895,48.54273,-3.55120,47.97842,8.67280,50.31249,9.74351,55.87496"
        elif area == "GS_BalticSea":
            area = "polygon=6.42936234081148,58.06933188935258,8.279936182852992,56.60995884195779,9.77463043988601,53.640768085976816,21.58983266214966,53.725076403620875,30.735937996854545,59.86677430413451,28.60066048680588,61.31689046459479,22.08806408116112,60.36326141464659,22.83541120967695,62.70289239987201,27.426257856280074,65.61608475872717,22.47953162466976,66.52536961810017,19.73925882010812,64.24418831137172,15.860171343522126,61.28270990295982,15.860171343522126,56.76631664113353,13.511366082469124,56.25577473222242,10.87785715341036,60.64361921650416,8.42228801685593,58.629547789876256,6.42936234081148,58.06933188935258"
        elif area == "GS_SouthChinaSea":
            area = "polygon=98.86609661997,8.761756775260409,96.16575957545422,4.8984661910612886,102.51949379784571,-1.9254572721213634,100.45453017556935,-10.366906480390398,123.08970834284287,-11.61431933913046,125.47235867623976,-16.466767514033222,133.65279148756986,-13.860707843374854,136.11486349874792,-12.778732842985974,138.73577886548395,-9.349684246097908,139.76826067662364,-5.807512090082426,137.14734530988608,-2.877669323455109,138.4975138321447,35.63428101232233,133.09683974311162,35.11621506468174,128.56980410965576,37.231645155575706,125.55178035401951,40.44749626033976,121.18358807612458,41.823204815858105,117.6890342538083,40.80915702051942,116.5771307648904,37.168383431979905,120.70705800944455,29.06545223449251,114.35332378705306,23.58737830287147,106.25231265350266,22.050118028915463,104.425614064564,18.323806173034896,108.31727627577902,13.127339365868266,104.82272245346422,12.11983013962174,99.18378333109047,14.8995938250698,98.86609661997,8.761756775260409"
        else:
            parsed = False

        if (area[0:5] == "bbox=") & (area.count(",") == 3):
            parsed = True

        if (area[0:8] == "polygon=") & (area.count(",") >= 7):
            parsed = True

        if (
            (area[0:4] == "lon=")
            & (area.count("&lat=") == 1)
            & (area.count("&radius=") == 1)
        ):
            parsed = True

        if not parsed:
            raise Exception(f"Failed to parse area {area}")
        return area

    def parse_date(self, date):
        """is the date accepted by the altimetry api? 
        
        Arguments:
            date -- datetime or string in format yyyymmdd are accepted
        
        Raises:
            TypeError: if neither string nor datetime
        
        Returns:
            date -- as yyyymmdd string
        """
        if type(date) == datetime:
            date = date.strftime("%Y%m%d")
        elif type(date) == str:
            if date is not "":
                self._validate_date_str(date)
        else:
            raise TypeError(f"Date should be either str or datetime")
        return date

    def _validate_date_str(self, date_text):
        try:
            datetime.strptime(date_text, "%Y%m%d")
        except ValueError:
            raise ValueError(
                f"Date format {date_text} is incorrect, should be YYYY-MM-DD"
            )

    def validate_quality_filter(self, qa):
        """is the quality filter string allowed by the altimetry api
        
        Arguments:
            qa -- allowed strings: '', 'qual_swh', 'qual_wind', 'qual_combined'
        
        Raises:
            Exception: if filter is unknown
        
        Returns:
            qa -- the accepted string
        """
        if qa == "nofilter":
            return ""
        elif (
            (qa == "")
            | (qa == "dhi_combined")
            | (qa == "qual_swh")
            | (qa == "qual_wind_speed")
        ):
            return qa
        else:
            raise Exception(
                f"Filter {qa} is unknown (dhi_combined, qual_swh, qual_wind_speed)"
            )

    def get_altimetry_from_api(self, url_query, access_token=None):
        """Request data from altimetry api
        
        Arguments:
            url_query -- query build with create_altimetry_query() 
        
        Keyword Arguments (optional):
            api_key -- if not already set at initialization then supply
        
        Raises:
            APIAuthenticationFailed: if api key is wrong
        
        Returns:
            dataframe -- with altimetry data
        """
        if access_token == None:
            access_token = self.API_token
            if access_token == None:
                self.API_token = self.get_access_token()  
                access_token = self.API_token

        t_start = time.time()
        r = requests.get(
            url_query,             
            headers={"authorization": "bearer {access_token}".format(**access_token)}
            )
        if r.status_code == 401:
            raise APIAuthenticationFailed
        r.raise_for_status()
        response_data = r.json()
        df = pd.read_csv(
            response_data["download_url"], parse_dates=True, index_col="date"
        )

        elapsed = time.time() - t_start
        nrecords = len(df)
        if nrecords > 0:
            print(
                f"Succesfully retrieved {nrecords} records from API in {elapsed:.2f} seconds"
            )
        else:
            print("No data retrieved!")

        return df

    # def get_altimetry_from_api_old(self, url_query, max_tries=10):
    # """
    # Given a URL to retrieve data from, fetch the data and return it in a Pandas DataFrame.

    # Parameters
    # ----------
    # url_query : str
    #     URL where data is located.
    # max_tries : int
    #     Maximum number of times to try to fetch the data until the response is 200.

    # Returns
    # -------
    # Pandas DataFrame
    #     DataFrame containing the data located at the URL specified in `url_query`.

    # Raises
    # -------
    # DownloadURLGenerationFailed
    #     If the number of maximal tries is exceeded without getting a response status equal to 200.
    # """
    # response = requests.get(url_query, headers={'Authorization': 'Bearer '+ self.api_key})

    # if response.status_code is 401:
    #     raise APIAuthenticationFailed('API respond: 401 unauthorized access - maybe get new access key')

    # n_tries = 0
    # while response.status_code is not 200:
    #     response = requests.get(url_query)
    #     n_tries = n_tries + 1
    #     if n_tries > max_tries:
    #         raise DownloadDataFromAPIFailed(f'Max number of retries {max_tries} reached')

    # response_dict = eval(response.text)
    # data = requests.get(response_dict['download_url'])
    # df = pd.read_csv(StringIO(data.text))
    # return df

    def parse_satellite_list(self, satellites):
        """
        Parse a list of satellite names into an argument string to pass as part of a URL query.

        Parameters
        ----------
        satellites : Union[List[str], str]
            List of short or long names of satellite to include, an empty string, or the string 'sentinels' to specify
                the two sentinel satellites 3a and 3b.

        Returns
        -------
        str
            String representing argument specifying which satellites to retrieve data from.

        Raises
        --------
        InvalidSatelliteName
            If a string that is not the empty string, 'sentinels', or part of the following lists is passed:
            ['TOPEX', 'Poseidon', 'Jason-1', 'Envisat', 'Jason-2', 'SARAL', 'Jason-3',
                                'Geosat', 'GFO', 'ERS-1', 'ERS-2', 'CryoSat-2', 'Sentinel-3A', 'Sentinel-3B']
            ['tx', 'ps', 'j1', 'n1', 'j2', 'sa', 'j3', 'gs', 'g1', 'e1', 'e2', 'c2', '3a', '3b']
        """
        if satellites is "sentinels":
            satellites = ["3a", "3b"]

        if satellites is "":
            return ""

        satellite_long_names = [
            "TOPEX",
            "Poseidon",
            "Jason-1",
            "Envisat",
            "Jason-2",
            "SARAL",
            "Jason-3",
            "Geosat",
            "GFO",
            "ERS-1",
            "ERS-2",
            "CryoSat-2",
            "Sentinel-3A",
            "Sentinel-3B",
        ]

        satellite_short_names = [
            "tx",
            "ps",
            "j1",
            "n1",
            "j2",
            "sa",
            "j3",
            "gs",
            "g1",
            "e1",
            "e2",
            "c2",
            "3a",
            "3b",
        ]

        satellite_dict = dict(zip(satellite_long_names, satellite_short_names))

        satellite_strings = []
        for satellite in satellites:
            if satellite in satellite_short_names:
                satellite_strings.append(satellite)
            elif satellite in satellite_long_names:
                satellite_strings.append(satellite_dict[satellite])
            else:
                raise InvalidSatelliteName(
                    "Invalid satellite name passed: " + satellite
                )
        satellite_string = ",".join(satellite_strings)
        return "satellites=" + satellite_string
