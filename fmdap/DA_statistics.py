import numpy as np
import matplotlib.pyplot as plt



class InnoStats:
    """Calculate and store innovation statistics     
    """
    rms = 0.0
    minval = 0.0
    maxval = 0.0
    bias = 0.0

    def __init__(self, dfi):
        if dfi is not None:
            self.calc(dfi)
            

    def calc(self, dfi):
        rms = 0.0
        minval = 0.0
        maxval = 0.0
        bias = 0.0

        # TODO: calc

        self.rms = rms
        self.minval = minval
        self.maxval = maxval
        self.bias = bias
        return rms, minval, maxval, bias


class Statistics:
    # Calculates BIAS, RMS, Min/Max Errors of innovation (y-Hx). assume constant
    # observation operators H and m.
    dff = None    # df with *forecast* cols: ens1, ens2, ..., ensM, obs
    dfa = None    # df with *analysis* cols: 
    dfi = None    # 

    def __init__(self, dff, dfa):
        super().__init__()

    def calc_innovation_statistics(self, dfi):
        return False

    def plot_rank_histogram(self):
        #chrono = stats.HMM.t

        fig, ax = plt.subplots(24, (6,3), loc="3313")
        ax.set_title('(Mean of marginal) rank histogram (_a)')
        ax.set_ylabel('Freq. of occurence\n (of truth in interval n)')
        ax.set_xlabel('ensemble member index (n)')

#   #has_been_computed = \
#       hasattr(stats,'rh') and \
#       not all(stats.rh.a[-1]==array(np.nan).astype(int))

#   if has_been_computed:
#     ranks = stats.rh.a[chrono.maskObs_BI]
#     Nx    = ranks.shape[1]
#     N     = stats.config.N
#     if not hasattr(stats,'w'):
#       # Ensemble rank histogram
#       integer_hist(ranks.ravel(),N)
#         return False

    def integer_hist(self,E,N,centrd=False,weights=None,**kwargs):
        """Histogram for integers."""
        ax = plt.gca()
        rnge = (-0.5,N+0.5) if centrd else (0,N+1)
        ax.hist(E,bins=N+1,range=rnge,density=True,weights=weights,**kwargs)
        ax.set_xlim(rnge)


    def freshfig(self,num,figsize=None,*args,**kwargs):
        """Create/clear figure, as in:
        >>> fig, ax = suplots(*args,**kwargs)

        With the modification that:
        - If the figure does not exist: create it.
        This allows for figure sizing -- even with mpl backend MacOS.
        Also place figure.
        - Otherwise: clear figure.
        Avoids closing/opening so as to keep pos and size.
        """
        exists = plt.fignum_exists(num)

        fig = plt.figure(num=num,figsize=figsize)
        fig.clf()

        _, ax = plt.subplots(num=fig.number,*args,**kwargs)
        return fig, ax
