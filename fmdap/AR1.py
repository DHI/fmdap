import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess


def phi_to_halflife(phi, dt=1.0):
    """Convert the AR(1) propagation parameter phi to half-life"""
    rho = dt / (np.log2(1 / phi))
    return rho


def halflife_to_phi(rho, dt=1):
    """Convert half-life to the AR(1) propagation parameter phi"""
    phi = 0.5 ** (dt / rho)
    # st_dev = st_dev * np.sqrt(1 - phi**2)
    return phi


def _fit_AR1(data):
    """Fit AR(1) model to the data"""
    mod = ARIMA(data, order=(1, 0, 0))
    res = mod.fit()
    return res


def estimate_AR1_halflife(df):
    """Estimate AR(1) half-life of columns in DataFrame"""
    if not isinstance(df, (pd.DataFrame, pd.Series)):
        raise TypeError(f"Type {type(df)} not supported (DataFrame/Series)")
    data = df.to_numpy()

    if data.squeeze().ndim == 2:
        return np.array([estimate_AR1_halflife(df[col]) for col in df])

    res = _fit_AR1(df)
    phi = res.params[1]

    try:
        dt = pd.Timedelta(df.index.freq).total_seconds()
        # dt = pd.infer_freq(df.index).delta.seconds
    except:
        dt = 1.0
    return phi_to_halflife(phi, dt=dt)


def simulate_AR1(*, phi=None, halflife=None, st_dev=1, index=None, n_sample=1000):

    dt = 1
    if index is not None:
        assert isinstance(index, pd.DatetimeIndex)
        dt = index.freq.delta.seconds
        n_sample = len(index)

    if phi is None:
        if halflife is None:
            raise ValueError("Either 'phi' or 'halflife' must be provided")
        phi = halflife_to_phi(halflife, dt=dt)

    st_dev = st_dev * np.sqrt(1 - phi ** 2)
    ar_coeff = np.array([1, -phi])
    AR1_process = ArmaProcess(ar_coeff, ma=None)
    simulated_AR1 = st_dev * AR1_process.generate_sample(nsample=n_sample)

    if index is not None:
        df = pd.DataFrame(simulated_AR1, index=index)
        df.columns = ["AR1"]
        return df
    else:
        return simulated_AR1
