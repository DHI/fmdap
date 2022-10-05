import random
import numpy as np
from mikeio.spatial.utils import dist_in_meters
from scipy.optimize import curve_fit


def _pairwise_distance(ec, is_geo=False):
    """Calculate all pairwise distances for coordinates"""
    n = len(ec)
    d = np.zeros((n, n))
    for j in range(n):
        d[j, 0:j] = dist_in_meters(ec[0:j, :], ec[j, :], is_geo=is_geo)
        d[0:j, j] = d[j, 0:j]
    return d


def _dist_corrcoef_to_array(dist, cc):
    """Take unique values from distance and corrcoef"""
    dd = np.triu(dist, 1).flatten()
    cc = np.triu(cc, 1).flatten()
    ok = np.logical_and(dd > 0, ~np.isnan(cc))
    return dd[ok], cc[ok]   


def get_distance_and_corrcoef(dfs, item=0, n_sample=100):
    """Calculate pairwise distances and correlations from sample of point in dfs file"""
    # assert isinstance(dfs, Dfsu)

    elem_ids = None
    if n_sample is not None and n_sample < dfs.n_elements:
        n_sample = min(n_sample, dfs.n_elements)
        elem_ids = random.sample(range(0, dfs.n_elements), n_sample)

    ec = dfs.element_coordinates[elem_ids, :2]
    dd = _pairwise_distance(ec, is_geo=dfs.is_geo)

    ds = dfs.read(items=[item], elements=elem_ids)[0]
    cc = np.corrcoef(ds.values.T)

    return _dist_corrcoef_to_array(dd, cc)


def gaussian(x, spatial_corr):
    """Gaussian distance function with value 1 at 0 and zero mean"""
    return np.exp(-0.5 * (x / spatial_corr) ** 2)


def fit_gaussian(x, values, max_dist=None):
    """Fit the Gaussian distance function to data"""
    spatial_corr_guess = np.mean(x)
    if max_dist:
        idx = x < max_dist
        x = x[idx].copy()
        values = values[idx].copy()
    popt, _ = curve_fit(gaussian, x, values, spatial_corr_guess)
    return popt[0]


#
