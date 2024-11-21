# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 12:44:25 2021

@author: ASUS
"""

import xarray as xr
import glob

import numpy as np
import xarray as xr

# Libraries for plotting and visualising data
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

pr_list1 = glob.glob('D:/descarga _esgf/esgf_descarga/1/*.nc')
pr_list2 = glob.glob('D:/descarga _esgf/esgf_descarga/2/*.nc')
pr_list3 = glob.glob('D:/descarga _esgf/esgf_descarga/3/*.nc')
pr_list4 = glob.glob('D:/descarga _esgf/esgf_descarga/4/*.nc')
pr_list5 = glob.glob('D:/descarga _esgf/esgf_descarga/5/*.nc')
pr_list6 = glob.glob('D:/descarga _esgf/esgf_descarga/6/*.nc')
pr_list7 = glob.glob('D:/descarga _esgf/esgf_descarga/7/*.nc')
pr_list8 = glob.glob('D:/descarga _esgf/esgf_descarga/8/*.nc')
pr_list9 = glob.glob('D:/descarga _esgf/esgf_descarga/9/*.nc')
pr_list10 = glob.glob('D:/descarga _esgf/esgf_descarga/10/*.nc')

CanESM2 = xr.open_mfdataset(pr_list1, chunks={'time':'500MB'})
print(CanESM2)
print(CanESM2['pr'])
CanESM2['pr'].data
CanESM2['time'].data############ visualizar el tiempo
#print(CCCma['pr'].max(),CCCma['pr'].min())

CanESM2 = xr.open_mfdataset(pr_list1)
CSIRO = xr.open_mfdataset(pr_list2)
ICHEC = xr.open_mfdataset(pr_list3)
IPSL = xr.open_mfdataset(pr_list4)
MIROC = xr.open_mfdataset(pr_list5)
MOHC = xr.open_mfdataset(pr_list6)
NorESM1 = xr.open_mfdataset(pr_list7)
NOAA = xr.open_mfdataset(pr_list8)
CanESM2_SAM20 = xr.open_mfdataset(pr_list9)
MIROC_SAM20= xr.open_mfdataset(pr_list10)


d = CSIRO ['pr']*86400
d1 = CSIRO['pr']*86400
d2 = ICHEC['pr']*86400

yearly_mean = d.groupby('time.year').mean('time')
yearly_mean1 = d1.groupby('time.year').mean('time')
yearly_mean2 = d2.groupby('time.year').mean('time')

ref = yearly_mean.where((yearly_mean.year > 2071) & (yearly_mean.year < 2080), drop=True)
ref1 = yearly_mean1.where((yearly_mean1.year > 2071) & (yearly_mean1.year < 2080), drop=True)
ref2 = yearly_mean2.where((yearly_mean2.year > 2071) & (yearly_mean2.year < 2080), drop=True)
ref_mean = ref.mean(dim="year")
ref_mean1 = ref1.mean(dim="year")
ref_mean2 = ref2.mean(dim="year")
ref_mean.plot()

ref_global = ref_mean.mean(["rlon", "rlat"])
ref_global1 = ref_mean1.mean(["rlon", "rlat"])
ref_global2 = ref_mean2.mean(["rlon", "rlat"])

# global mean for annual data
yearly_mean_global = yearly_mean.mean(["rlon", "rlat"])
yearly_mean_global1 = yearly_mean1.mean(["rlon", "rlat"])
yearly_mean_global2 = yearly_mean2.mean(["rlon", "rlat"])


anomalies_global = yearly_mean_global - ref_global
anomalies_global1 = yearly_mean_global1 - ref_global1
anomalies_global2 = yearly_mean_global2 - ref_global2

fig = plt.figure(figsize=(8,5))
ax = plt.subplot()
ax.set_ylabel('t2m anomaly (Kelvin)')
ax.set_xlabel('year')
ax.plot(anomalies_global.year, anomalies_global, color='green', label='Global anomalies')
ax.plot(anomalies_global1.year, anomalies_global1, color='red', label='Global anomalies')
ax.plot(anomalies_global2.year, anomalies_global2, color='yellow', label='Global anomalies')
#ax.plot(mean_line.year, mean_line, color='red', linestyle='dashed', label='Mean anomaly 1981-2019')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)
ax.set_title('Global anomalies of t2m from 2010 to 2040')
