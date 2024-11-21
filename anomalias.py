# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 01:05:14 2021

@author: ASUS
"""
import numpy as np
import xarray as xr

# Libraries for plotting and visualising data
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
t2m = "D:/descarga _esgf/CIMP5_2006_2100/tasmax_Amon_CanESM2_rcp85_r1i1p1_200601-210012.nc"

# Create Xarray Dataset
ds = xr.open_dataset(t2m)
ds

# Create Xarray Data Array
da = ds['tasmax']

yearly_mean = da.groupby('time.year').mean('time')

#Calcule la temperatura media para un período de referencia de 1981 a 2010:
ref = yearly_mean.where((yearly_mean.year > 2006) & (yearly_mean.year < 2020), drop=True)
ref_mean = ref.mean(dim="year")
ref_mean.plot()

#Ahora trazaremos una serie de tiempo global de anomalías de temperatura anual, definidas como desviaciones en la temperatura de la media de referencia.

#Calcule la media global para el período de referencia (2010,2020) y para los datos anuales de 1979 a 2019:
    
    # global mean for reference period
ref_global = ref_mean.mean(["lon", "lat"])

# global mean for annual data
yearly_mean_global = yearly_mean.mean(["lon", "lat"])

anomalies_global = yearly_mean_global - ref_global

# Create a dashed horizontal line to show where the reference temperature lies
mean_line = xr.DataArray(0.0, coords=[('year', np.arange(2010,2040))])

fig = plt.figure(figsize=(8,5))
ax = plt.subplot()
ax.set_ylabel('t2m anomaly (Kelvin)')
ax.set_xlabel('year')
ax.plot(anomalies_global.year, anomalies_global, color='green', label='Global anomalies')
ax.plot(mean_line.year, mean_line, color='red', linestyle='dashed', label='Mean anomaly 1981-2019')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)
ax.set_title('Global anomalies of t2m from 2010 to 2040')

#Trazar la temperatura media del Ártico
#Primero tenemos que crear un subconjunto para el Círculo Polar Ártico (por encima de los 66 ° 33'N de latitud, o 66.55 en grados decimales)

arctic = da.where(da.lat >= 66.55, drop=True)

#Calcule la temperatura media de 2 metros en el período 1979 a 2019:
    
arctic_mean_2006_to_2100 = arctic.mean(dim="time")

# create the figure panel 
fig = plt.figure(figsize=(5,5))
# create the map using the cartopy Orthographic projection, selecting the North Pole
ax = plt.subplot(1,1,1, projection=ccrs.Orthographic(central_latitude=90.0))
# add coastlines
ax.coastlines()
# compute a circle in axes coordinates, which we can use as a boundary for the map.
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
# set boundary
ax.set_extent([-180,180, 66.55,90], crs=ccrs.PlateCarree())
ax.set_boundary(circle, transform=ax.transAxes)
# provide a title
ax.set_title('Mean Arctic t2m for period 2006 to 2100')
# plot t2m
pp = plt.pcolormesh(arctic_mean_2006_to_2100 .lon, arctic_mean_2006_to_2100.lat,
                    arctic_mean_2006_to_2100 , cmap='viridis', transform=ccrs.PlateCarree())
# add colourbar
cbar = plt.colorbar(pp)
cbar.set_label(label='t2m (Kelvin)')

#https://github.com/siljeci/wekeo-jupyter-lab/blob/master/climate/WEkEO_climate_training_2_climatologies.ipynb