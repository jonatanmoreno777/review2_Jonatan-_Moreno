# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 21:42:21 2022

@author: ASUS
"""
# Disable warnings for data download via API
import urllib3 
urllib3.disable_warnings()
# CDS API
import cdsapi
# Libraries for working with multidimensional arrays
import numpy as np
import xarray as xr

# Libraries for plotting and visualising data
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature

# Create Xarray Dataset
ds = xr.open_dataset('D:/R/RAIN4PE_daily_0.1d_1981_2015_v1.0.nc')
ds
yearly_mean = ds.groupby('time.year').mean(keep_attrs=True)

ref = yearly_mean.where((yearly_mean.year > 1985) & (yearly_mean.year < 2016), drop=True)

ref_mean = ref.mean(dim="year", keep_attrs=True)

t2m_2010 = yearly_mean.sel(year=2010)

anom_2010 = t2m_2010 - ref_mean

# create the figure panel and the map using the Cartopy PlateCarree projection
fig, ax = plt.subplots(1, 1, figsize = (16, 8), subplot_kw={'projection': ccrs.PlateCarree()})

# Plot the data
im = plt.pcolormesh(anom_2010.Longitude, anom_2010.Latitude, anom_2010, cmap='RdBu_r', vmin=-3, vmax=3) 

# Set the figure title, add lat/lon grid and coastlines
ax.set_title('Near-surface air temperature anomaly for 2016 (with respect to 1991-2020 mean)', fontsize=16)
ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--') 
ax.coastlines(color='black')
ax.set_extent([-74, -72, -15, -12], crs=ccrs.PlateCarree())

# Specify the colourbar
cbar = plt.colorbar(im,fraction=0.05, pad=0.04)
cbar.set_label('temperature anomaly') 


clim_period = ds.sel(time=slice('1991-01-01', '2010-12-01'))
clim_month = clim_period.groupby('time.month').mean()

weights = np.cos(np.deg2rad(clim_month.Latitude))
weights.name = "weights"
clim_month_weighted = clim_month.weighted(weights)

mean = clim_month_weighted.mean(["Longitude", "Latitude"])
mean.pcp.plot()

clim_std = clim_period.groupby('time.month').std()

weights = np.cos(np.deg2rad(clim_std.Latitude))
weights.name = "weights"
clim_std_weighted = clim_std.weighted(weights)

std = clim_std_weighted.mean(["Longitude", "Latitude"])

fig, ax = plt.subplots(1, 1, figsize = (12, 6))

ax.plot(mean.month, mean, color='blue', label='mean')
ax.fill_between(mean.month, (mean + std), (mean - std), alpha=0.1, color='green', label='+/- 1 SD')

ax.set_title('European monthly climatology of 2m temperature (1991 to 2020)')
ax.set_ylabel('° C')
ax.set_xlabel('month')
ax.set_xlim(1,12)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)
ax.grid(linestyle='--')

https://github.com/stewartchrisecmwf/training/blob/main/C3S_climatology.ipynb




