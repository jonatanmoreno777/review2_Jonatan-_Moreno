# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 08:00:58 2022

@author: ASUS
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import argparse
import salem

RAIN4PE_day = xr.open_dataset("D:/R/RAIN4PE_daily_0.1d_1981_2015_v1.0.nc")

print(RAIN4PE_day)
RAIN4PE_day.variables.keys()
clim = RAIN4PE_day["pcp"].mean("time" , keep_attrs=True)
print(RAIN4PE_day.Longitude.attrs)
print(RAIN4PE_day.Latitude.attrs)

map_corte = clim.sel(Longitude=np.arange(-25,50),Latitude=np.arange(0.40,0.5), method="nearest")
map_corte.plot()

print((clim.max()),(clim.min()))
sam20=RAIN4PE_day["pcp"].groupby('time.season').mean('time', keep_attrs=True)


sam20.plot.contourf(x='Longitude', y='Latitude', col='season')
print((sam20.max()),(sam20.min()))
cont_levels = np.around(np.concatenate([np.arange(0, 1.1, 0.1), 
                                        np.arange(2, 25, 2)]),
                        decimals=1)

sam20.plot.contourf(x='Longitude', y='Latitude', col='season',cbar_kwargs={'label': clim.units}, 
                       col_wrap=4, levels=cont_levels, cmap='jet')

#grafica para una fecha especifica
RAIN4PE_day['pcp']
pr = RAIN4PE_day['pcp']
one_day = pr.sel(time='1981-01-03')
one_day.plot(robust=True)

import geopandas as gpd
import salem    #pip install scipy, joblib
from matplotlib.patches import Rectangle
shp = gpd.read_file("D:/T_JONA/TESIS_PISCO/DEPARTAMENTOS.shp")
shp3 = gpd.read_file("D:/T_JONA/TESIS_PISCO/cachi_wg84_R.shp")
crs_str = shp3.geometry.crs
print(shp)  
shp1 = shp.loc[shp['DEPARTAMEN'] == 'AYACUCHO']

ds = RAIN4PE_day.salem.subset(shape=shp, margin=450)#################### margin: tamaño de figura
dsr = sam20.salem.subset(shape=shp, margin=450)
dsr = dsr.salem.roi(shape=shp).plot.contourf(col="season", col_wrap=2,cbar_kwargs={'label': clim.units}, 
                                              levels=np.arange(-9, 25, 0.2), cmap='gnuplot2',hatches=['//'])

bottomright = dsr.axes[-1, -1]
bottomright.annotate("bottom right", (-75, -10), color= 'red')
plt.draw()

#plot promedio mensual
month_=RAIN4PE_day["pcp"].groupby('time.month').mean('time', keep_attrs=True)
print((month_.max()),(month_.min()))
print((sam20.max()),(sam20.min()))
ds = RAIN4PE_day.salem.subset(shape=shp, margin=450)#################### margin: tamaño de figura
dsr = sam20.salem.subset(shape=shp, margin=450)

dsr = dsr.salem.roi(shape=shp).plot.contourf(col="season", col_wrap=4,cbar_kwargs={'label': clim.units}, 
                                              levels=np.arange(-9, 25, 0.5), cmap='gnuplot2',hatches=['///'])
#corte cachi
ds = RAIN4PE_day.salem.subset(shape=shp1, margin=2)
dsr = sam20.salem.subset(shape=shp1, margin=2)
dsr = dsr.salem.roi(shape=shp1).plot.contourf(col="month", col_wrap=4,cbar_kwargs={'label': clim.units}, 
                                              levels=np.arange(0, 26, 0.5), cmap='gnuplot2',hatches=['///'])#ojo con levels

dsr = dsr.salem.roi(shape=shp1).plot.contourf(col="season", col_wrap=2, cbar_kwargs={'label': clim.units},#Ojo:correr de nuevo si sale error, ojito con levels:variacion en r pp(mm/day) averiguar
                                              levels=np.arange(-9, 25, 0.2), cmap='gnuplot2',hatches=['///'])#

import regionmask

pr_rain4pe = RAIN4PE_day.sel(Longitude=slice(-13,-74), Latitude= slice(-13,-74), time= slice('1981-01-01', '2015-12-31'))
peru = regionmask.Regions(np.array(shp.geometry))
mask= peru.mask(pr_rain4pe.isel(time=0), lat_name='Latitude', lon_name='Longitude')
pr_mask= pr_rain4pe.where(mask==0)

pr_mask.pcp.mean(dim='time').plot







