import numpy as np
import pandas as pd
import datetime

from mikeio import Dfs0

# TODO
# http://www.data-assimilation.net/Documents/sangomaDL6.14.pdf
# 1. Check whiteness of innovations
# 2. innovation histograms
# 3. Calc innovation statistics

class DiagnosticOutput:
    df = None
    time = None
    n_members = None
    #title = None
    #variable = None
    diag_type = 0
    nupdates = 0
    #_dfs = None
    #iforecast = np.zeros(0, dtype=bool)
    #ianalysis = np.zeros(0, dtype=bool)
    #inoupdate = np.zeros(0, dtype=bool)

    def __init__(self, filename=None):
        if filename is not None:
            self.read(filename)


    def read(self, filename):
        """Read diagnostic output dfs0 file, determine type and store as data frame
        
        Arguments:
            filename -- path to the dfs0 file
        """
        self.df = Dfs0(filename).to_dataframe()
        self.time = self.df.index.to_pydatetime()
        self._find_diag_type()


    def get_total_forecast(self): 
        """Get a data frame containing no-update and forecast values
        """
        iforecast, _, inoupdate = self.idx_at_updates()
        return self.df.iloc[iforecast|inoupdate]


    def get_total_analysis(self): 
        """Get a data frame containing no-update and analysis values
        """
        _, ianalysis, inoupdate = self.idx_at_updates()
        return self.df.iloc[ianalysis|inoupdate]


    def _find_diag_type(self, df=None):
        """Determine diagnostic type based on item names 
        
        Keyword Arguments:
            df -- data frame  (default: self)
        
        Raises:
            Exception: if None of the three diagnostic types could be identified
        
        Returns:
            diag_type -- diagnostic type (1, 2 or 3)
        """
        if df is None:
            df=self.df

        cols = list(df.columns)
        if cols[-1][0:10].lower() == 'mean state':
            diag_type = 2   
        elif (cols[-1][0:11].lower() == 'measurement') & (cols[-2][0:10].lower() == 'mean state'):
            diag_type = 1
        elif (cols[0][0:18] == 'points assimilated'):
            diag_type = 3
        else:
            raise Exception(f'Diagnostic type could not be determined - based on item names: {cols}')

        self.diag_type = diag_type
        return diag_type


    def idx_at_updates(self, df=None): 
        """Find index of updates in data frame
        
        Returns:
            iforecast -- index before updates (forecast)
            ianalysis -- index after updates (analysis)
            inoupdate -- index when there were no update
        """
        if df is None:
            df=self.df    

        time = df.index.to_pydatetime()
        nt   = len(time)
        dt = np.diff(time)
        ii = (dt==datetime.timedelta(0))   # find repeated datetimes

        if len(ii) == None:
            print('No updates were found in diagnostic file')
            return None, None, None

        iforecast = np.zeros(nt, dtype=bool)
        iforecast[0:-1] = ii
        ianalysis = np.zeros(nt, dtype=bool)
        ianalysis[1:] = ii
        inoupdate = (iforecast|ianalysis) != True
        nupdates = len(iforecast[iforecast==True])
        return iforecast, ianalysis, inoupdate


    def get_iforecast_from_ianalysis(self, ianalysis):
        nt   = len(ianalysis)
        iforecast = np.zeros(nt, dtype=bool)
        iforecast[0:-1] = ianalysis[1:]
        return iforecast


    def get_increments(self):
        """Determine the all increments 
        
        Returns:
            df_increment -- a dataframe containing all increments
        """
        iforecast, ianalysis, _ = self.idx_at_updates()            
        
        state_items = [i for i in list(self.df.columns) if i.startswith('State ')]        

        dff = self.df[state_items].iloc[iforecast]
        dfa = self.df[state_items].iloc[ianalysis]
        df_increment = dfa.subtract(dff)
        return df_increment


    def get_all_increments_as_array(self):
        """Determine the all increments and return as array
        
        Returns:
            increments -- a column vector containing all increments
        """
        df_increments = self.get_increments()
        return df_increments.values.reshape(-1,1)


    def get_mean_increments(self):
        """Determine the mean increments 
        
        Returns:
            df_increment -- a dataframe containing the mean increments
        """
        iforecast, ianalysis, _ = self.idx_at_updates()
        
        mean_state_item = [i for i in list(self.df.columns) if i.lower().startswith('mean state')]

        dff = self.df[mean_state_item].iloc[iforecast]
        dfa = self.df[mean_state_item].iloc[ianalysis]
        df_increment = dfa.subtract(dff)
        return df_increment
