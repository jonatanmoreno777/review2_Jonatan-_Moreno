# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 07:58:30 2022

@author: ASUS
"""

import numpy as np
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import gamma, norm
from scipy.signal import detrend
import xarray as xr

"""
module for bias corrections. 
Available methods include:
- basic_quantile
- modified quantile
- gamma_mapping
- normal_mapping 
"""

__version__ = "0.4"

def quantile_correction(obs_data, mod_data, sce_data, modified=True):
    cdf = ECDF(mod_data)
    p = cdf(sce_data) * 100
    cor = np.subtract(*[np.nanpercentile(x, p) for x in [obs_data, mod_data]])
    if modified:
        mid = np.subtract(*[np.nanpercentile(x, 50) for x in [obs_data, mod_data]])
        g = np.true_divide(*[np.nanpercentile(x, 50) for x in [obs_data, mod_data]])

        iqr_obs_data = np.subtract(*np.nanpercentile(obs_data, [75, 25]))
        iqr_mod_data = np.subtract(*np.nanpercentile(mod_data, [75, 25]))

        f = np.true_divide(iqr_obs_data, iqr_mod_data)
        cor = g * mid + f * (cor - mid)
        return sce_data + cor
    else:
        return sce_data + cor


def gamma_correction(
    obs_data, mod_data, sce_data, lower_limit=0.1, cdf_threshold=0.9999999
):
    obs_raindays, mod_raindays, sce_raindays = [
        x[x >= lower_limit] for x in [obs_data, mod_data, sce_data]
    ]
    obs_gamma, mod_gamma, sce_gamma = [
        gamma.fit(x) for x in [obs_raindays, mod_raindays, sce_raindays]
    ]

    obs_cdf = gamma.cdf(np.sort(obs_raindays), *obs_gamma)
    mod_cdf = gamma.cdf(np.sort(mod_raindays), *mod_gamma)
    sce_cdf = gamma.cdf(np.sort(sce_raindays), *sce_gamma)

    obs_cdf[obs_cdf > cdf_threshold] = cdf_threshold
    mod_cdf[mod_cdf > cdf_threshold] = cdf_threshold
    sce_cdf[sce_cdf > cdf_threshold] = cdf_threshold

    obs_cdf_intpol = np.interp(
        np.linspace(1, len(obs_raindays), len(sce_raindays)),
        np.linspace(1, len(obs_raindays), len(obs_raindays)),
        obs_cdf,
    )

    mod_cdf_intpol = np.interp(
        np.linspace(1, len(mod_raindays), len(sce_raindays)),
        np.linspace(1, len(mod_raindays), len(mod_raindays)),
        mod_cdf,
    )

    obs_inverse, mod_inverse, sce_inverse = [
        1.0 / (1.0 - x) for x in [obs_cdf_intpol, mod_cdf_intpol, sce_cdf]
    ]

    adapted_cdf = 1 - 1.0 / (obs_inverse * sce_inverse / mod_inverse)
    adapted_cdf[adapted_cdf < 0.0] = 0.0

    initial = (
        gamma.ppf(np.sort(adapted_cdf), *obs_gamma)
        * gamma.ppf(sce_cdf, *sce_gamma)
        / gamma.ppf(sce_cdf, *mod_gamma)
    )

    mod_frequency = 1.0 * mod_raindays.shape[0] / mod_data.shape[0]
    sce_frequency = 1.0 * sce_raindays.shape[0] / sce_data.shape[0]

    days_min = len(sce_raindays) * sce_frequency / mod_frequency

    expected_sce_raindays = int(min(days_min, len(sce_data)))

    sce_argsort = np.argsort(sce_data)
    correction = np.zeros(len(sce_data))

    if len(sce_raindays) > expected_sce_raindays:
        initial = np.interp(
            np.linspace(1, len(sce_raindays), expected_sce_raindays),
            np.linspace(1, len(sce_raindays), len(sce_raindays)),
            initial,
        )
    else:
        initial = np.hstack(
            (np.zeros(expected_sce_raindays - len(sce_raindays)), initial)
        )

    correction[sce_argsort[:expected_sce_raindays]] = initial
    # correction = pd.Series(correction, index=sce_data.index)
    return correction


def normal_correction(obs_data, mod_data, sce_data, cdf_threshold=0.9999999):
    obs_len, mod_len, sce_len = [len(x) for x in [obs_data, mod_data, sce_data]]
    obs_mean, mod_mean, sce_mean = [x.mean() for x in [obs_data, mod_data, sce_data]]
    obs_detrended, mod_detrended, sce_detrended = [
        detrend(x) for x in [obs_data, mod_data, sce_data]
    ]
    obs_norm, mod_norm, sce_norm = [
        norm.fit(x) for x in [obs_detrended, mod_detrended, sce_detrended]
    ]

    obs_cdf = norm.cdf(np.sort(obs_detrended), *obs_norm)
    mod_cdf = norm.cdf(np.sort(mod_detrended), *mod_norm)
    sce_cdf = norm.cdf(np.sort(sce_detrended), *sce_norm)

    obs_cdf = np.maximum(np.minimum(obs_cdf, cdf_threshold), 1 - cdf_threshold)
    mod_cdf = np.maximum(np.minimum(mod_cdf, cdf_threshold), 1 - cdf_threshold)
    sce_cdf = np.maximum(np.minimum(sce_cdf, cdf_threshold), 1 - cdf_threshold)

    sce_diff = sce_data - sce_detrended
    sce_argsort = np.argsort(sce_detrended)

    obs_cdf_intpol = np.interp(
        np.linspace(1, obs_len, sce_len), np.linspace(1, obs_len, obs_len), obs_cdf
    )
    mod_cdf_intpol = np.interp(
        np.linspace(1, mod_len, sce_len), np.linspace(1, mod_len, mod_len), mod_cdf
    )
    obs_cdf_shift, mod_cdf_shift, sce_cdf_shift = [
        (x - 0.5) for x in [obs_cdf_intpol, mod_cdf_intpol, sce_cdf]
    ]

    obs_inverse, mod_inverse, sce_inverse = [
        1.0 / (0.5 - np.abs(x)) for x in [obs_cdf_shift, mod_cdf_shift, sce_cdf_shift]
    ]

    adapted_cdf = np.sign(obs_cdf_shift) * (
        1.0 - 1.0 / (obs_inverse * sce_inverse / mod_inverse)
    )
    adapted_cdf[adapted_cdf < 0] += 1.0
    adapted_cdf = np.maximum(np.minimum(adapted_cdf, cdf_threshold), 1 - cdf_threshold)

    xvals = norm.ppf(np.sort(adapted_cdf), *obs_norm) + obs_norm[-1] / mod_norm[-1] * (
        norm.ppf(sce_cdf, *sce_norm) - norm.ppf(sce_cdf, *mod_norm)
    )

    xvals -= xvals.mean()
    xvals += obs_mean + (sce_mean - mod_mean)

    correction = np.zeros(sce_len)
    correction[sce_argsort] = xvals
    correction += sce_diff - sce_mean
    # correction = pd.Series(correction, index=sce_data.index)
    return correction


class BiasCorrection(object):
    def __init__(self, obs_data, mod_data, sce_data):
        self.obs_data = obs_data
        self.mod_data = mod_data
        self.sce_data = sce_data

    def correct(
        self, method="modified_quantile", lower_limit=0.1, cdf_threshold=0.9999999
    ):
        if method == "gamma_mapping":
            corrected = gamma_correction(
                self.obs_data,
                self.mod_data,
                self.sce_data,
                lower_limit=lower_limit,
                cdf_threshold=cdf_threshold,
            )
        elif method == "normal_mapping":
            corrected = normal_correction(
                self.obs_data, self.mod_data, self.sce_data, cdf_threshold=cdf_threshold
            )
        elif method == "basic_quantile":
            corrected = quantile_correction(
                self.obs_data, self.mod_data, self.sce_data, modified=False
            )

        elif method == "modified_quantile":
            corrected = quantile_correction(
                self.obs_data, self.mod_data, self.sce_data, modified=True
            )

        else:
            raise Exception("Specify correct method for bias correction.")

        self.corrected = pd.Series(corrected, index=self.sce_data.index)
        return self.corrected


class XBiasCorrection(object):
    def __init__(self, obs_data, mod_data, sce_data, dim="time"):
        self.obs_data = obs_data
        self.mod_data = mod_data
        self.sce_data = sce_data
        self.dim = dim

    def correct(
        self,
        method="modified_quantile",
        lower_limit=0.1,
        cdf_threshold=0.9999999,
        vectorize=True,
        dask="parallelized",
        **apply_ufunc_kwargs
    ):
        dtype = self._set_dtype()
        dim = self.dim
        if method == "gamma_mapping":
            corrected = xr.apply_ufunc(
                gamma_correction,
                self.obs_data,
                self.mod_data,
                self.sce_data,
                vectorize=vectorize,
                dask=dask,
                input_core_dims=[[dim], [dim], [dim]],
                output_core_dims=[[dim]],
                output_dtypes=[dtype],
                kwargs={"lower_limit": lower_limit, "cdf_threshold": cdf_threshold},
                **apply_ufunc_kwargs
            )
        elif method == "normal_mapping":
            corrected = xr.apply_ufunc(
                normal_correction,
                self.obs_data,
                self.mod_data,
                self.sce_data,
                vectorize=vectorize,
                dask=dask,
                input_core_dims=[[dim], [dim], [dim]],
                output_core_dims=[[dim]],
                output_dtypes=[dtype],
                kwargs={"cdf_threshold": cdf_threshold},
                **apply_ufunc_kwargs
            )
        elif method == "basic_quantile":
            corrected = xr.apply_ufunc(
                quantile_correction,
                self.obs_data,
                self.mod_data,
                self.sce_data,
                vectorize=vectorize,
                dask=dask,
                input_core_dims=[[dim], [dim], [dim]],
                output_core_dims=[[dim]],
                kwargs={"modified": False},
                **apply_ufunc_kwargs
            )

        elif method == "modified_quantile":
            corrected = xr.apply_ufunc(
                quantile_correction,
                self.obs_data,
                self.mod_data,
                self.sce_data,
                vectorize=vectorize,
                dask=dask,
                input_core_dims=[[dim], [dim], [dim]],
                output_core_dims=[[dim]],
                kwargs={"modified": True},
                **apply_ufunc_kwargs
            )

        else:
            raise Exception("Specify correct method for bias correction.")
        self.corrected = corrected
        return self.corrected

    def _set_dtype(self):
        aa = self.mod_data
        if isinstance(aa, xr.Dataset):
            dtype = aa[list(aa.data_vars)[0]].dtype
        elif isinstance(aa, xr.DataArray):
            dtype = aa.dtype
        return dtype
    
    
from bias_correction import XBiasCorrection
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt


obs_data = np.random.randn(2*365, 34, 65)
model_data = np.random.randn(2*365, 34, 65)
sce_data = np.random.randn(365, 34, 65)

lat = range(34)
lon = range(65)

obs_data = xr.DataArray(obs_data, dims=['time','lat','lon'], \
                        coords=[pd.date_range('2010-01-01', '2011-12-31', freq='D'), lat, lon])
model_data = xr.DataArray(model_data, dims=['time','lat','lon'], \
                          coords=[pd.date_range('2010-01-01', '2011-12-31', freq='D'), lat, lon])
sce_data = xr.DataArray(sce_data, dims=['time','lat','lon'], \
                        coords=[pd.date_range('2019-01-01', '2019-12-31', freq='D'), lat, lon])

# combining dataarrays to form dataset with same timelength otherwise correction will give error
ds = xr.Dataset({'model_data':model_data,'obs_data':obs_data, 'sce_data':sce_data})
ds['sce_data']

bc = XBiasCorrection(ds['obs_data'], ds['model_data'], ds['sce_data'])
# df1 = bc.correct(method='modified_quantile')
# df2 = bc.correct(method='basic_quantile')
df3 = bc.correct(method='gamma_mapping')

bc.sce_data.sel(time='2019-12-31').plot(figsize=(14, 5), robust=True)
plt.show()

df3.sel(time='2019-12-31').plot(figsize=(14, 5), robust=True)
plt.show()
