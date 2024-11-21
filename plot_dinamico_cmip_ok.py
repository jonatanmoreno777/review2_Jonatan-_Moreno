# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 09:44:36 2021

@author: ASUS
"""
import cartopy.crs as ccrs
import matplotlib.animation as animation
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature


ds = xr.open_dataset("D:/descarga _esgf/CIMP5_2006_2100/pr_Amon_CanESM2_rcp85_r1i1p1_200601-210012.nc", decode_times=True)
hp_shp = Reader("D:/T_JONA/TESIS_PISCO/DEPARTAMENTOS.shp")
tas = ds.pr
#<xarray.DataArray 'pr' (time: 1140, lat: 64, lon: 128)##1140 meses- tramas por long y lat
fig = plt.figure()##(tas[0,:,:]*86400).plot.contourf(levels=60) convertir de  kg m-2 s-1 a mm/dia
#Ahora que hemos calculado la climatología, queremos convertir las unidades de kg m-2 s-1 a algo con lo que estemos un poco más familiarizados como mm día-1.
#Para ello, considere que 1 kg de agua de lluvia repartida en 1 m2 de superficie tiene 1 mm de espesor y que hay 86400 segundos en un día. Por lo tanto, 1 kg m-2 s-1 = 86400 mm día-1.
(tas[0,:,:]*86400).plot.contourf(levels=60)#(tas[0,:,:]*30).plot.contourf(levels=60) convertir de  kg m-2 s-1 a mm/mes suponiendo un promedio de 30 dias 


fig = plt.figure()

# explicitly set up axis with projection
ax = plt.axes(projection=ccrs.PlateCarree())

# add coastlines
ax.coastlines(linewidths=0.5)

(tas[0,:,:]*86400*30).plot.contourf(ax=ax,  # * 86400   (tas[0,:,:]*86400)
                           transform=ccrs.PlateCarree(),
                           levels=300,
                           cmap="jet",
                           cbar_kwargs={
                               "orientation": "horizontal",
                               "label": "$pr [mm/mes]$"}
                            )

# configure axes
ax.set_xlabel("")
#ax.set_xticks(np.arange(-180, 181, 30))

ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
#ax1.add_feature(cfeature.BORDERS, linestyle=":")
#ax1.add_feature(cfeature.LAKES, alpha =0.5)
ax.add_feature(cfeature.RIVERS)
#ax1.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle='dotted')
grid_lines = ax.gridlines(draw_labels=True, color="black", alpha=0.02, linestyle="--")
grid_lines.xformatter = LONGITUDE_FORMATTER
grid_lines.yformatter = LATITUDE_FORMATTER

ax.gridlines()

## desea agregar un shp
#ax.add_feature(ShapelyFeature(hp_shp.geometries(),ccrs.PlateCarree(), edgecolor='red'), facecolor='none')
    

ax.set_ylabel("")
#ax.set_yticks(np.arange(-90, 91, 30));

from matplotlib.animation import FuncAnimation

def update(i):
    tas[i, :, :].plot.contourf(ax=ax,###tiempo i
                       transform=ccrs.PlateCarree(),
                       levels=60,
                       cmap="jet",
                       add_colorbar=False
                        )
    # configure axes
    
    ax.set_xlabel("")
    #ax.set_xticks(np.arange(-180, 181, 30))

    ax.set_ylabel("")
    #ax.set_yticks(np.arange(-90, 91, 30))
    
    # Set plot title
    ax.set_title("Precipitación - CMIP5 (pr) - mes " + str(i+1));
    
    
# runs the animation
anim = FuncAnimation(fig, update, frames=36)#### 12 tramas - meses-primeros meses 

#anim = animation.FuncAnimation(fig, update, frames=1140, interval=200)

# Uncomment this line to save the created animation
anim.save('D:/descarga _esgf/animate_j9.gif', writer='pillow', fps=5, dpi=300)

# deseo ver el gif dinamico in spyder?
from matplotlib import rc
plt.rcParams['animation.html'] ="jshtml"
rc('animation', html='html5')#['html5', 'jshtml', 'none']
anim


#clase pandas y numpy ok
#https://github.com/AtmaMani/pyChakras/blob/3f71f53c0d90e463ade4547575166d403620ca93/udemy_ml_bootcamp/Python-for-Data-Analysis/Pandas/DataFrames.ipynb
#https://github.com/AtmaMani/pyChakras/blob/3f71f53c0d90e463ade4547575166d403620ca93/udemy_ml_bootcamp/Python-for-Data-Analysis/NumPy/NumPy%20Arrays.ipynb


