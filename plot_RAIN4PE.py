# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 06:50:32 2022

@author: ASUS
"""

# Libraries for working with multi-dimensional arrays
import numpy as np
import xarray as xr
import pandas as pd

from IPython.display import HTML

# Libraries for plotting and visualising data
import matplotlib.path as mpath
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import animation

from datetime import datetime

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature


ds_global_aod = xr.open_dataset('D:/R/RAIN4PE_daily_0.1d_1981_2015_v1.0.nc')
ds_global_aod

aod550 = ds_global_aod['pcp']
aod550
aod550.time
aod_unit = aod550.units
aod_long_name = aod550.long_name


matrix = np.array([[210, 214, 234],
                   [167, 174, 214],
                   [135, 145, 190],
                   [162, 167, 144],
                   [189, 188, 101],
                   [215, 209, 57],
                   [242, 230, 11],
                   [243, 197, 10],
                   [245, 164, 8],
                   [247, 131, 6],
                   [248, 98, 5],
                   [250, 65, 3],
                   [252, 32, 1],
                   [254, 0, 0]])


# Multiplication number
n = 18

# 'cams' is an initial empty colourmap, to be filled by the colours provided in 'matrix'.
cams = np.ones((253, 4))

# This loop fills in the empty 'cams' colourmap with each of the 14 colours in 'matrix'
# multiplied by 'n', with the first row left as 1 (white). Note that each colour value is 
# divided by 256 to normalise the colour range from 0 (black) to 1 (white). 
for i in range(matrix.shape[0]):
    cams[((i*n)+1):(((i+1)*n)+1),:] = np.array([matrix[i,0]/256, matrix[i,1]/256, matrix[i,2]/256, 1])

# The final color map is given by 'camscmp', which uses the Matplotlib class 'ListedColormap(Colormap)'
# to generate a colourmap object from the list of colours provided by 'cams'.
camscmp = ListedColormap(cams)



time_index =  10

# Initiate the matplotlib figure
fig = plt.figure(figsize=(16,8))
ax = plt.subplot(1,1,1, projection=ccrs.PlateCarree())

# Plotting function with pcolormesh
im = plt.pcolormesh(aod550.Longitude, aod550.Latitude, aod550[time_index,:,:],
                    cmap=camscmp, transform=ccrs.PlateCarree())

# Add additional mapping features
ax.coastlines(color='black')
ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
cbar = plt.colorbar(im,fraction=0.046, pad=0.05)
cbar.set_label(aod_unit)

# Set a title of the plot
ax.set_title(aod_long_name + ' - ' + str(aod550.time[time_index].values)+'\n', fontsize=16)
import matplotlib as mpl
ds_global_aod.pcp[:].plot(yincrease=True, x='Longitude', y='Latidude', xlim=(-70, -74), ylim=(-12, -18), cmap=mpl.cm.Blues)

def visualize_pcolormesh(data_array, longitude, latitude, projection, color_scale, unit, long_name, vmin, vmax, 
                         set_global=True, lonmin=-180, lonmax=180, latmin=-90, latmax=90):
    """ 
    Visualizes a xarray.DataArray with matplotlib's pcolormesh function.
    
    Parameters:
        data_array(xarray.DataArray): xarray.DataArray holding the data values
        longitude(xarray.DataArray): xarray.DataArray holding the longitude values
        latitude(xarray.DataArray): xarray.DataArray holding the latitude values
        projection(str): a projection provided by the cartopy library, e.g. ccrs.PlateCarree()
        color_scale(str): string taken from matplotlib's color ramp reference
        unit(str): the unit of the parameter, taken from the NetCDF file if possible
        long_name(str): long name of the parameter, taken from the NetCDF file if possible
        vmin(int): minimum number on visualisation legend
        vmax(int): maximum number on visualisation legend
        set_global(boolean): optional kwarg, default is True
        lonmin,lonmax,latmin,latmax(float): optional kwarg, set geographic extent is set_global kwarg is set to 
                                            False

    """
    fig=plt.figure(figsize=(20, 10))

    ax = plt.axes(projection=projection)
   
    img = plt.pcolormesh(longitude, latitude, data_array, 
                        cmap=plt.get_cmap(color_scale), transform=ccrs.PlateCarree(),
                        vmin=vmin,
                        vmax=vmax,
                        shading='auto')

    ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=1)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1)

    if (projection==ccrs.PlateCarree()):
        ax.set_extent([lonmin, lonmax, latmin, latmax], projection)
        gl = ax.gridlines(draw_labels=True, linestyle='--')
        gl.top_labels=False
        gl.right_labels=False
        gl.xformatter=LONGITUDE_FORMATTER
        gl.yformatter=LATITUDE_FORMATTER
        gl.xlabel_style={'size':14}
        gl.ylabel_style={'size':14}

    if(set_global):
        ax.set_global()
        ax.gridlines()

    cbar = fig.colorbar(img, ax=ax, orientation='horizontal', fraction=0.04, pad=0.1)
    cbar.set_label(unit, fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    ax.set_title(long_name, fontsize=20, pad=20.0)

    return fig, ax


# Setting the initial state:
# 1. Define figure for initial plot
fig, ax = visualize_pcolormesh(data_array=aod550[0,:,:],
                               longitude=aod550.Longitude, 
                               latitude=aod550.Latitude,
                               projection=ccrs.PlateCarree(), 
                               color_scale=camscmp, 
                               unit=aod_unit,
                               long_name=aod_long_name + ' '+ str(aod550.time[0].data),
                               vmin=0,
                               vmax=40, 
                               lonmin=aod550.Longitude.min(), 
                               lonmax=aod550.Longitude.max(), 
                               latmin=aod550.Latitude.min(), 
                               latmax=aod550.Latitude.max(),
                               set_global=False)

frames = 31

def draw(i):
    img = plt.pcolormesh(aod550.Longitude, 
                         aod550.Latitude, 
                         aod550[i,:,:], 
                         cmap=camscmp, 
                         transform=ccrs.PlateCarree(),
                         vmin=0,
                         vmax=2,
                         shading='auto')
    
    ax.set_title(aod_long_name + ' '+ str(aod550.time[i].data), fontsize=20, pad=20.0)
    return img


def init():
    return fig


def animate(i):
    return draw(i)

ani = animation.FuncAnimation(fig, animate, frames, interval=500, blit=False,
                              init_func=init, repeat=True)

HTML(ani.to_html5_video())
plt.close(fig)





