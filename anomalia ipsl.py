# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 07:19:10 2021

@author: ASUS
"""

import xarray as xr
import glob
import os
import pandas as pd
import requests
from tqdm import tqdm
import xarray as xr
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns
import sys

#pp_ipsl = glob.glob('D:/descarga _esgf/esgf_descarga/IPSL/pr/*.nc')
tmpmax_ipsl = glob.glob('D:/descarga _esgf/esgf_descarga/IPSL/tmax/*.nc')
tmpmin_ipsl = glob.glob('D:/descarga _esgf/esgf_descarga/IPSL/tmin/*.nc')


#print(CCCma['pr'].max(),CCCma['pr'].min())


df_temp = xr.open_mfdataset(tmpmax_ipsl, engine="netcdf4")
#df_temp1 = xr.open_mfdataset(tmpmin_ipsl, engine="netcdf4")


df_temp_deg=df_temp.tasmax-273.15
df_temp_deg.attrs = df_temp.tasmax.attrs
df_temp_deg["units"] = "deg C"
df_temp_deg


interan_temp=df_temp.groupby('time.year').mean() #Annual mean
df = interan_temp.tasmax.to_dataframe()
dd=df.tasmax.groupby('year').mean() # Yearly mean
clim_temp=df.loc[2046:2060,].tasmax.groupby('year').mean().mean() # climatology mean
clim_sd=df.loc[2046:2060,].tasmax.groupby('year').mean().std()# climatology standard deviation

temp_anom=(dd-clim_temp)/clim_sd # anomaly compuatation

x = temp_anom.index[1:54]
y = temp_anom[1:54]

from scipy.interpolate import UnivariateSpline

s = UnivariateSpline(x, y, k=3, s=1)
y_spline_1 = s(x)
s = UnivariateSpline(x, y, k=3, s=0.4)
y_spline_2 = s(x)

fig = plt.figure(figsize=(10,5))
cols=['red' if x <0 else 'blue' for x in temp_anom]
plt.bar(temp_anom.index[1:54],temp_anom[1:54],color=cols)##35 años de lluvia
plt.xlabel('Años'),plt.ylabel('Anomalía')
plt.title('Anomalía de temperatura estandarizada', loc='left', fontsize=15)
plt.title('2046-2100',loc='right', fontsize=15)
#plt.plot(x, y_spline_1, "m-", lw=2)
#plt.plot(x, y_spline_2, "m--", lw=2)
plt.show()


fig =plt.figure(figsize=(10, 5))
plt.bar(x, y, 0.4, lw=0)
plt.plot(x, y_spline_1, "m-", lw=2)
plt.plot(x, y_spline_2, "m--", lw=2)
plt.xlim([1948, 2015])
plt.xlabel("Year")
plt.ylabel("Temperatuer anomaly in Degrees Celsius")
plt.title("Mean Temperature Anomalies Relative to 1950-1980 baseline")
plt.show()

https://github.com/certik/climate

https://github.com/certik/climate/blob/master/Temperature%20Fits%20Reproduction.ipynb
################################################################################

df_temp_deg=df_temp1.tasmin-273.15
df_temp_deg.attrs = df_temp.tasmax.attrs
df_temp_deg["units"] = "deg C"
df_temp_deg


interan_temp=df_temp1.groupby('time.year').mean() #Annual mean
df = interan_temp.tasmin.to_dataframe()
dd=df.tasmin.groupby('year').mean() # Yearly mean
clim_temp=df.loc[2046:2070,].tasmin.groupby('year').mean().mean() # climatology mean
clim_sd=df.loc[2046:2070,].tasmin.groupby('year').mean().std()# climatology standard deviation

temp_anom1=(dd-clim_temp)/clim_sd # anomaly compuatation

fig = plt.figure(figsize=(10,5))
cols=['red' if x <0 else 'blue' for x in temp_anom1]
plt.bar(temp_anom1.index[1:54],temp_anom[1:54],color=cols)##35 años de lluvia
plt.xlabel('Years'),plt.ylabel('Anomaly')
plt.title('Temperature standardised anomaly', loc='left', fontsize=15)
plt.title('2046-2100',loc='right', fontsize=15)
plt.show()
















