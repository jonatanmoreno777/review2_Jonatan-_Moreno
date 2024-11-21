# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 11:40:50 2022

@author: ASUS
"""

import numpy as np
import seaborn as sns
import xarray as xr
import dask
from dask.distributed import performance_report
from dask.distributed import Client
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import cftime
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as feature

from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Times New Roman']},size=12)# 'Computer Modern'
rc('text', usetex=False)

#https://github.com/RichardScottOZ/dask-era5/blob/main/notebook/cmip6_zarr.ipynb
#https://github.com/RichardScottOZ/uncover-ml/blob/master/notebooks/diagnostics.ipynb
ds = xr.open_dataset("D:/T_JONA/TESIS_PISCO/Entrada/Pisco_Pp/Precday.nc")#ds = xr.open_dataset("D:/T_JONA/TESIS_PISCO/Entrada/Pisco_Pp/PISCOpm.nc", engine="netcdf4",decode_times=False)#PARA MENSUAL
ds = xr.open_dataset("D:/2020/Leer PISCO/PISCOpd.nc.nc", engine="netcdf4")
ds = xr.open_dataset("D:/T_JONA/TESIS_PISCO/Entrada/Pisco_Pp/PISCOpm.nc", engine="netcdf4",decode_times=False, use_cftime=True)#PARA MENSUAL


# Look at the netCDF representation
ds.info()
# variables are in our dataset
ds.data_vars
# select one variable and pick the first entry along the first axis (time)
ds.variable[0]
# Plot one timestep
ds.variable[0].plot()

#############plot para un año especifico#############
precip = ds['variable']
precip
precip_first = precip[0]########3primer año
precip_first.plot()

print((precip.min()),(precip.max()))##########para vmax

ax = plt.axes(projection=ccrs.PlateCarree())
precip_first_mm = precip_first#precip_first_mm = precip_first*86400.
precip_first_mm.plot(ax=ax, vmin=0., vmax=50.)  # we tell the command to use the geographical axis just defined
ax.coastlines(color='w')

############################################################
clim = ds["variable"].mean("z" , keep_attrs=True)

plt.rcParams['figure.figsize'] = (7, 5)
clim.plot(vmin=-5, vmax=35, cmap='hsv')#ojo con rangos vmin y vmax
plt.xlabel('$Longitude$')
plt.ylabel('$Latitude$')


###########suma de las precipitaciones en todo el año en mm
sam20=ds['variable'].groupby('z.month').sum(dim="z")
# let's plot all months with one cool command!
sam20.plot.contourf(row='month', col_wrap=4, vmin=0., vmax=150., alpha=0.5, levels=20, cmap="hsv")

###########promedio de las precipitaciones en todo el año en mm
sam20=ds['variable'].groupby('z.month').mean(dim="z")
# let's plot all months with one cool command!
sam20.plot.contourf(row='month', col_wrap=4, vmin=0., vmax=150., alpha=0.5, levels=20, cmap="hsv")
############################################################################################################
import geopandas as gpd
import salem
shp = gpd.read_file("D:/T_JONA/TESIS_PISCO/DEPARTAMENTOS.shp")

dset = ds.salem.subset(shape=shp, margin=450)#################### margin: tamaño de figura
dsr = clim.salem.subset(shape=shp, margin=450)
dsr = dsr.salem.roi(shape=shp).plot.contourf(cmap='jet',
                                             levels=np.arange(0, 48, 0.2), hatches=['///'])

####DISTRIBUCION DE LA PRECIPITACION#################
pr_data = clim.data[np.logical_not(np.isnan(clim.data))]
sns.distplot(pr_data)
plt.title('Global land precipitation distribution')
plt.xlabel('mm/day')
plt.show()

# dataset dimensions
ds.dims
# dataset coordinates
ds.coords
# dataset global attributes
ds.attrs
# Extract the sst variable/datarray
ds["variable"]  # Equivalent to ds.variable
# The actual (numpy) array data
ds.variable.data
# dataarray/variable dimensions
ds.variable.dims
# datarray/variable coordinates
ds.variable.coords
# dataarray/variable attributes
ds.variable.attrs

# calculates the mean along the time dimension
temp_mean = ds['variable'].mean(dim='z')
temp_mean = temp_mean.compute()

temp_mean.plot(figsize=(20, 10))
plt.title('pr')
temp_std = ds['variable'].std(dim='z')
temp_std = temp_std.compute()
temp_std.plot(figsize=(20, 10))
plt.title('2015-2100 Standard Deviation 2-m Air Temperature')

# location coordinates

###################################  DESCARGA PP_PISCO   ######################
locs = [
    {'name': 'Est_V1', 'longitude': -74.063, 'latitude': -12.958},#r-130-741   1
    {'name': 'Est_V2', 'longitude': -74.375, 'latitude': -12.958},#r-130-744    2
    {'name': 'Est_V3', 'longitude': -74.063, 'latitude': -13.270},#r-133-741   3
    {'name': 'Est_V4', 'longitude': -74.375, 'latitude': -13.270},#r-133-744   4
    {'name': 'Est_V5', 'longitude': -74.688, 'latitude': -13.270},#r-133-747  5
    {'name': 'Est_V6', 'longitude': -74.063, 'latitude': -13.582},#r-136-741  6
    {'name': 'Est_V7', 'longitude': -74.375, 'latitude': -13.582},#r-136-744  7
    {'name': 'Est_V8', 'longitude': -74.688, 'latitude': -13.582},#r-136-747   8
]    
    
ds_locs = xr.Dataset()
air_temp_ds = ds

# interaccionar a través de las ubicaciones y crear un conjunto de datos
# que contiene los valores de PP para cada ubicación
for l in locs:
    name = l['name']
    longitude = l['longitude']
    latitude = l['latitude']
    var_name = name

    ds2 = air_temp_ds.sel(longitude=longitude, latitude=latitude, method='nearest')

    longitude_attr = '%s_longitude' % name
    latitude_attr = '%s_latitude' % name

    ds2.attrs[longitude_attr] = ds2.longitude.values.tolist()
    ds2.attrs[latitude_attr] = ds2.latitude.values.tolist()
    ds2 = ds2.rename({'variable' : var_name}).drop(('latitude', 'longitude'))

    ds_locs = xr.merge([ds_locs, ds2], compat='override')

ds_locs.data_vars

df_f = ds_locs.to_dataframe()
ax = df_f.plot(figsize=(20, 8), title="Precipitación", grid=2)
ax.set(xlabel='Date', ylabel='( mm/día )')
#plt.axhline(y=20, c='gray', ls=':')
#plt.axvline(x=1995, c='gray', ls='--')
#plt.axvline(x=1989, c='gray', ls='--')
leg = plt.legend(facecolor='black', framealpha=0.8)
for text in leg.get_texts():
    plt.setp(text, color = 'w')
ax = plt.gca()
ax.set_facecolor('#faf5e6')
plt.text(0.01, 0.98 ,"Cuenca Cachi", fontsize=14, transform=ax.transAxes, verticalalignment='top', color='blue')
plt.text(0.88, 0.98 ,"PISCO", fontsize=14, transform=ax.transAxes, verticalalignment='top', color='red')


df_f.to_csv("D:/Clase R en Hdrología/picopy5_2016.csv", index=True)
df_f.describe()


###################################  DESCARGA TMP_PISCO   ######################

locs = [
    {'name': 'Est_V1', 'X': -74.063, 'Y': -12.958},#r-130-741   1
    {'name': 'Est_V2', 'X': -74.375, 'Y': -12.958},#r-130-744    2
    {'name': 'Est_V3', 'X': -74.063, 'Y': -13.270},#r-133-741   3
    {'name': 'Est_V4', 'X': -74.375, 'Y': -13.270},#r-133-744   4
    {'name': 'Est_V5', 'X': -74.688, 'Y': -13.270},#r-133-747  5
    {'name': 'Est_V6', 'X': -74.063, 'Y': -13.582},#r-136-741  6
    {'name': 'Est_V7', 'X': -74.375, 'Y': -13.582},#r-136-744  7
    {'name': 'Est_V8', 'X': -74.688, 'Y': -13.582},#r-136-747   8
]    
    
##############https://github.com/NCristea/eScienceIncubator/blob/master/process_wrf_data.ipynb######ejemplo descargar series de tiempo de .nc
#https://github.com/mattijn/pynotebook/blob/24e70e8bb62a8487e64951f5876df2af4cda062b/2017/2017-01-17%20Display%20Timeseries%20MODIS%20MCD15A3H.006-checkpoint.ipynb
# convert westward longitudes to degrees east
for l in locs:############################OJO NO ES NECESARIO PARA PISCO
    if l['longitude'] < 0:
        l['longitude'] = 360 + l['longitude']
locs

ds_locs = xr.Dataset()
air_temp_ds = ds

# interaccionar a través de las ubicaciones y crear un conjunto de datos
# que contiene los valores de temperatura para cada ubicación
for l in locs:
    name = l['name']
    X = l['X']
    Y = l['Y']
    var_name = name

    ds2 = air_temp_ds.sel(X=X, Y=Y, method='nearest')

    X_attr = '%s_X' % name
    Y_attr = '%s_Y' % name

    ds2.attrs[X_attr] = ds2.X.values.tolist()
    ds2.attrs[Y_attr] = ds2.Y.values.tolist()
    ds2 = ds2.rename({'tmin' : var_name}).drop(('Y', 'X'))

    ds_locs = xr.merge([ds_locs, ds2], compat='override')

ds_locs.data_vars

df_f = ds_locs.to_dataframe()
df_f.to_csv("D:/Clase R en Hdrología/picopytminok_2016.csv", index=True)
df_f.describe()

df_f = ds_locs.to_dataframe()
ax = df_f.plot(figsize=(20, 8), title="Temperatura mínima", grid=2)
ax.set(xlabel='Date', ylabel='( C° )')
plt.axhline(y=0, c='red', ls=':')
#plt.axhline(y=10.0, c='red', ls=':')
#plt.axvline(x=1995, c='gray', ls='--')
#plt.axvline(x=1989, c='gray', ls='--')
leg = plt.legend(facecolor='black', framealpha=0.8)
for text in leg.get_texts():
    plt.setp(text, color = 'w')
ax = plt.gca()
ax.set_facecolor('#faf5e6')
plt.text(0.01, 0.98 ,"Cuenca Cachi", fontsize=14, transform=ax.transAxes, verticalalignment='top', color='blue')

ax = df_f.plot(figsize=(20, 8), title="pr", grid=1)
ax.set(xlabel='Date', ylabel='2-m Air Temperature (deg F)')
plt.show()

fig1 = df_f.plot(figsize=(15,4), subplots = True)

############plot para pr diaria###################
ds.plot.scatter(latitude=slice(-13.65, -12.75), longitude=slice(-74.9, -73.8)).mean(['latitude','longitude']).plot(figsize=(15,6), color='pink')
plt.grid()
#plt.axhline(y=0, color='k')
plt.xlim('1981-01-01', '2016-01-01')
plt.ylabel('curl taux')
#plt.ylim(ymin=-0.15,ymax=0.20)
plt.title("Zonal wind stress curl over same region as Arnold's figure");


#Windrose como un histograma apilado con resultados normalizados (mostrados en porcentaje)
from windrose import WindroseAxes 

ax = WindroseAxes.from_ax()
ax.bar(df_f.Est_V1, df_f.Est_V2, normed=True, opening=0.8, edgecolor='white')
ax.set_legend()
#Otra representación de histograma apilado, no normalizado, con límites de bins

ax = WindroseAxes.from_ax()
ax.box(df_f.Est_V1, df_f.Est_V2, bins=np.arange(0, 8, 1))
ax.set_legend()
import matplotlib.cm as cm
ax = WindroseAxes.from_ax()
ax.contour(df_f.Est_V1, df_f.Est_V2, bins=np.arange(0, 8, 1), cmap=cm.hot, lw=3)
ax.set_legend()


###########DESCARGA DE PP AREAL DE CUENCA##########################################33

#https://github.com/nzahasan/pyscissor/tree/master/notebooks
import pylab as pl
import fiona, numpy as np
from pyscissor import scissor
from netCDF4 import Dataset,num2date
import cartopy.crs as ccrs
from shapely.geometry import shape
from datetime import datetime as dt
from cartopy.feature import ShapelyFeature
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt

# read shapefile and netcdf file
ncf= Dataset("D:/T_JONA/TESIS_PISCO/Entrada/Pisco_Pp/Precday.nc")
sf = fiona.open("D:/T_JONA/TESIS_PISCO/cachi_wg84_R.shp")

lats =ncf.variables['latitude'][:]
lons =ncf.variables['longitude'][:] # increasing lat (reversed)
temp =ncf.variables['variable'][:] 
times=num2date(ncf.variables['z'][:],ncf.variables['z'].units,ncf.variables['z'].calendar)[:]
times_py=[dt(x.year,x.month,x.day,x.hour,x.minute) for x in times]

#calculo de max min de variables
print((lons.min()),(lons.max()))
print((temp.min()),(temp.max()))
#Calcular cuadrícula de peso de máscara para shapefile
record = next(iter(sf))
shapely_shape = shape(record['geometry'])

pys = scissor(shapely_shape,lats,lons)
wg = pys.get_masked_weight()

#Trazar cuadrícula ponderada
# cartopy feature
cart_ft = ShapelyFeature([shapely_shape],ccrs.PlateCarree(),facecolor='none',edgecolor='cyan',linewidth=2)

# plot
fig = plt.figure(figsize=[12,5])
ax1 = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
ax1.set_global()
ax1.coastlines()
ax1.add_feature(cfeature.LAND)
ax1.add_feature(cfeature.OCEAN)
ax1.add_feature(cfeature.COASTLINE)
#ax1.add_feature(cfeature.BORDERS, linestyle=":")
#ax1.add_feature(cfeature.LAKES, alpha =0.5)
ax1.add_feature(cfeature.RIVERS)
#ax1.add_feature(cfeature.COASTLINE)
ax1.add_feature(cfeature.BORDERS, linestyle='dotted')
grid_lines = ax1.gridlines(draw_labels=True, color="black", alpha=0.2, linestyle="--")
grid_lines.xformatter = LONGITUDE_FORMATTER
grid_lines.yformatter = LATITUDE_FORMATTER
#plt.plot(xs, ys, "*b", ms=20, transform=ccrs.PlateCarree())
#plt.plot(xs, ys, transform=ccrs.PlateCarree())
ax1.gridlines()

#fig =pl.figure(figsize=(5,5))
#ax=pl.axes(projection=ccrs.PlateCarree())
plt.pcolormesh(lons,lats,wg,cmap='cividis_r',shading='nearest')
plt.colorbar()
ax1.set_extent([-74.9, -73.8, -13.65, -12.75], ccrs.PlateCarree())
ax1.add_feature(cart_ft)

#Trazar precipitacion enmascarada
# assign mask 
temp.mask=wg.mask

# plot masked temperature of first time step
fig = plt.figure(figsize=[12,5])
ax1 = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
ax1.set_global()
ax1.coastlines()
ax1.add_feature(cfeature.LAND)
ax1.add_feature(cfeature.OCEAN)
ax1.add_feature(cfeature.COASTLINE)
#ax1.add_feature(cfeature.BORDERS, linestyle=":")
#ax1.add_feature(cfeature.LAKES, alpha =0.5)
ax1.add_feature(cfeature.RIVERS)
#ax1.add_feature(cfeature.COASTLINE)
ax1.add_feature(cfeature.BORDERS, linestyle='dotted')
grid_lines = ax1.gridlines(draw_labels=True, color="black", alpha=0.2, linestyle="--")
grid_lines.xformatter = LONGITUDE_FORMATTER
grid_lines.yformatter = LATITUDE_FORMATTER
#plt.plot(xs, ys, "*b", ms=20, transform=ccrs.PlateCarree())
#plt.plot(xs, ys, transform=ccrs.PlateCarree())
ax1.gridlines()
plt.pcolormesh(lons,lats,temp[-1],cmap='plasma_r')
plt.colorbar()
ax1.add_feature(cart_ft)
ax1.set_extent([-74.9, -73.8, -13.65, -12.75], ccrs.PlateCarree())
pl.show()


#Serie temporal de datos promedio del área enmascarada
## create a zero array of tiempo
temp_avg = np.zeros(times.shape[0])

for t in range(times.shape[0]):
    temp_avg[t] = temp[t].mean()
    
# plot
pl.figure(figsize=(10,3))
pl.plot(times_py,temp_avg,'o-')


#####Descarga en formato .csv
import pandas as pd
## De numpy a pandas
df = pd.DataFrame(data = temp_avg , index=times_py)
#df.rename?
#df.rename({"0":"prday_promedio_cachi"}, axis = 1)
#https://www.youtube.com/watch?v=XXDK3f45EGs
#df.to_csv("D:/T_JONA/TESIS_PISCO/Entrada/Pisco_Pp/piscopy3.csv", index=True)

ax = df.plot(figsize=(20, 8), title="pr promedio diaria a nivel de cuenca", grid=1, color ="r")
ax.set(xlabel='$Date$', ylabel='pr promedio diaria a nivel de cuenca (mm/dia)')
plt.show()

#Serie de tiempo promedio ponderado del área enmascarada
#El promedio ponderado es útil para calcular la lluvia. Aquí la temperatura se utiliza con fines de demostración
w_temp_avg = np.zeros(times.shape[0])

for t in range(times.shape[0]):
    w_temp_avg[t] = np.average(temp[t],weights=wg)
    
# plot
pl.figure(figsize=(12,5))
pl.plot(times_py,w_temp_avg,'o-',label='Weighted Average')
pl.plot(times_py,temp_avg,'o-',label='Average')
pl.legend()

#https://github.com/nzahasan/pyscissor/blob/master/tools/nc2ts_by_shp.py######Ejecute este script para extraer series temporales reducidas de un
#netCDF bajo las regiones poligonales de un shapefile.



#https://joehamman.com/2013/10/12/plotting-netCDF-data-with-Python/

##############clastering#####################3
#https://github.com/rasbt/python-machine-learning-book-2nd-edition/blob/1b01e733d15a1808ebdb0e07e46dbb9cb1634323/code/ch11/ch11.ipynb
https://www.youtube.com/watch?v=7xHsRkOdVwo
https://github.com/iphysresearch/S_Dbw_validity_index/blob/master/Plot.ipynb

