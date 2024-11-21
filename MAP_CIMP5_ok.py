# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 10:17:54 2021

@author: ASUS
"""

# imports!
from mpl_toolkits.basemap import Basemap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt
import numpy as np, numpy
import cmocean
import argparse
import seaborn as sns
#import cdutil
import iris
import iris.plot as iplt
from netCDF4 import Dataset

dset = xr.open_dataset("D:/descarga _esgf/CIMP5_2006_2100/pr_Amon_CanESM2_rcp85_r1i1p1_200601-210012.nc")

#spyder 4

lat  = dset.variables['lat'][:]
lon  = dset.variables['lon'][:]
data = dset.variables['pr'][1,:,:]*86400#data = dset.variables['variable'][1,:,:]
dset.close()

print((data.min()),(data.max()))
clim = dset["pr"].mean("time" , keep_attrs=True)

#clim.data = clim.data * 86400
clim.attrs['units'] = 'mm/day'

# convert the lat/lon values to x/y projections.
fig = plt.figure(figsize=(10,8))

m=Basemap(projection='robin',resolution='c',lat_0=0,lon_0=0)

#m=Basemap(projection='mill',lat_ts=10,llcrnrlon=lon.min(), \
  #urcrnrlon=lon.max(),llcrnrlat=lat.min(),urcrnrlat=lat.max(), \
  #resolution='c')

x, y = m(*np.meshgrid(lon,lat))

m.pcolormesh(x,y,data,shading='flat',cmap=plt.cm.cool)
m.colorbar(location='right')

# Add a coastline and axis values.

m.drawcoastlines()
m.fillcontinents()
m.drawmapboundary()
m.drawparallels(np.arange(-90.,90.,30.),labels=[1,0,0,0])
m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1])
m.contourf(x,y,data[:],levels=np.arange(0, 13.5, 1.5),
                   extend='max',
                   cbar_kwargs={'label': clim.attrs['units']},
                   cmap="cool")

#plot points
nylat = -12.901; nylon = -74.324 
x,y = m(nylon, nylat)
m.plot(x, y, 'ro', markersize=9)
plt.title('$CIMP5 2006 2100/pr Amon CanESM2 rcp85 r1i1p1 200601-210012$')
plt.show()
#################################################################################

####map ortogonal
m = Basemap(projection='ortho',lat_0=-12.901,lon_0=-74.324,resolution='l')
lon2, lat2 = np.meshgrid(lon,lat)
x, y = m(lon2, lat2)
fig = plt.figure(figsize=(15,7))
#m.fillcontinents(color='gray',lake_color='gray')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,20.))
m.drawmeridians(np.arange(-180.,181.,20.))
m.drawmapboundary(fill_color='jet')
cs = m.contourf(x,y,data[:],cmap=plt.cm.get_cmap('jet'))
plt.title('')
#plt.colorbar();

# Agregue un punto que muestre la ubicación de Peru
nylat = -12.901; nylon = -74.324 
x, y = m(nylon,nylat)
nyc = plt.plot(x,y,'wo')
plt.setp(nyc,'markersize',.5,'markeredgecolor','r')


#npstere
m = Basemap(projection='npstere',boundinglat=60,lon_0=0,resolution='l')
x, y = m(lon2, lat2)
fig = plt.figure(figsize=(15,7))
m.fillcontinents(color='gray',lake_color='gray')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,20.))
m.drawmeridians(np.arange(-180.,181.,20.))
m.drawmapboundary(fill_color='white')
m.contourf(x,y,data[:],cmap=plt.cm.get_cmap('jet'))
plt.title('CIMP5_2006_2100/pr_Amon_CanESM2_rcp85_r1i1p1_200601-210012')
plt.colorbar();


#####plot
fig = plt.figure(figsize=(18,8))
m= Basemap(projection='robin',resolution='c',lat_0=0,lon_0=0)
m.bluemarble()

# Add some more info to the map
cstl = m.drawcoastlines(linewidth=.5)
meri = m.drawmeridians(np.arange(0,360,60), linewidth=.2, labels=[1,0,0,1], labelstyle='+/-', color='grey' ) 
para = m.drawparallels(np.arange(-90,90,30), linewidth=.2, labels=[1,0,0,1], labelstyle='+/-', color='grey')
boun = m.drawmapboundary(linewidth=0.5, color='grey')

#plot points
nylat = -12.901; nylon = -74.324 
x,y = m(nylon, nylat)
m.plot(x, y, 'ro', markersize=5)
plt.show()




map_ = Basemap(projection='robin',resolution='c',lat_0=0,lon_0=0)

# Add some more info to the map
cstl = map_.drawcoastlines(linewidth=.5)
meri = map_.drawmeridians(np.arange(0,360,60), linewidth=.2, labels=[1,0,0,1], labelstyle='+/-', color='grey' ) 
para = map_.drawparallels(np.arange(-90,90,30), linewidth=.2, labels=[1,0,0,1], labelstyle='+/-', color='grey')
boun = map_.drawmapboundary(linewidth=0.5, color='grey')


import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

accesscm2_pr_file = "C:/Users/ASUS/Desktop/data-carpentry/data/pr_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_201001-201412.nc"

dset = xr.open_dataset(accesscm2_pr_file)

clim = dset['pr'].mean('time', keep_attrs=True)

clim.data = clim.data * 86400
clim.attrs['units'] = 'mm/day'
print((clim.data.max()),(clim.data.min()))

fig = plt.figure(figsize=[12,5])
ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=180))
clim.plot.contourf(ax=ax,
                   levels=np.arange(0, 13.5, 1.5),
                   extend='max',
                   transform=ccrs.PlateCarree(),
                   cbar_kwargs={'label': clim.units},
                   cmap='viridis_r')
ax.coastlines()
plt.show()


import xarray as xr
import cartopy.crs as cartopy
import matplotlib.pyplot as plt
import numpy as np

accesscm2_pr_file = "C:/Users/ASUS/Desktop/data-carpentry/data/pr_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_201001-201412.nc"

dset = xr.open_dataset(accesscm2_pr_file)

clim = dset['pr'].mean('time', keep_attrs=True)

clim.data = clim.data * 86400
clim.attrs['units'] = 'mm/day'
print((clim.data.max()), (clim.data.min()))

fig = plt.figure(figsize=[12, 5])
ax = fig.add_subplot(111, projection=ccrs.Orthographic(central_longitude=-60, central_latitude=-18))  # Ajusta la central_latitude según tus necesidades
clim.plot.contourf(ax=ax,
                   levels=np.arange(0, 13.5, 1.5),
                   extend='max',
                   transform=ccrs.PlateCarree(),
                   cbar_kwargs={'label': clim.units},
                   cmap='.9')

# Agregar grillas
# Ajustar la densidad de las grillas
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlocator = plt.MaxNLocator(10)  # Ajustar la densidad en el eje x
gl.ylocator = plt.MaxNLocator(10)  # Ajustar la densidad en el eje y

# Agregar límites de la costa y el país
ax.coastlines(linewidth=0.8)
ax.add_feature(cartopy.feature.BORDERS, linewidth=0.5, linestyle='--', edgecolor='black')


