from statsmodels.tsa.arima.model import ARIMA

# from statsmodels.tsa.arima_process import ArmaProcess
import numpy as np
import pandas as pd


def phi_to_halflife(phi, dt=1):
    rho = dt / (np.log2(1 / phi))
    return rho


def halflife_to_phi(rho, dt=1):
    phi = 0.5 ** (dt / rho)
    # st_dev = st_dev * np.sqrt(1 - phi**2)
    return phi


def fit_AR1(data):
    # Fit an AR(1) model to the data

    mod = ARIMA(data, order=(1, 0, 0))
    res = mod.fit()
    return res


def estimate_AR1_halflife(df):
    if not isinstance(df, (pd.DataFrame, pd.Series)):
        raise TypeError(f"Type {type(df)} not supported (DataFrame/Series)")
    data = df.to_numpy()

    if data.squeeze().ndim == 2:
        return np.array([estimate_AR1_halflife(df[col]) for col in df])

    res = fit_AR1(df)
    phi = res.params[1]

    try:
        dt = df.index.freq.delta.seconds
        # dt = pd.infer_freq(df.index).delta.seconds
    except:
        dt = 1
    return phi_to_halflife(phi, dt=dt)
