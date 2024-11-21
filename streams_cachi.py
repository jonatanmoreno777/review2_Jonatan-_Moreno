# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 16:18:17 2021

@author: ASUS
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import geopandas as gpd
from pysheds.grid import Grid
import mplleaflet
from pyproj import CRS

from IPython.display import IFrame
#import gdal

grid = Grid.from_raster('D:/T_JONA/TESIS_PISCO/srtm_90_cg/dem_cachi_cg.tif', data_name='dem')

def plotFigure(data, label, cmap='Blues'):
    plt.figure(figsize=(12,10))
    plt.imshow(data, extent=grid.extent, cmap=cmap)
    # plt.imshow(data, cmap=cmap)

    plt.colorbar(label=label)
    plt.grid()
    
    
plotFigure(grid.dem, 'Elevation (m)')


depressions = grid.detect_depressions('dem')

# Plot pits
fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(depressions, cmap='cubehelix', zorder=1)
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.title('Depressions', size=14)


grid.fill_depressions(data='dem', out_name='flooded_dem')

flats = grid.detect_flats('flooded_dem')

# Plot flats
fig, ax = plt.subplots(figsize=(8,8))
plt.imshow(flats, cmap='cubehelix', zorder=1)
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.title('Flats', size=14)

grid.resolve_flats(data='flooded_dem', out_name='inflated_dem')

plotFigure(grid.inflated_dem, 'flats', 'cubehelix')

flats_test = grid.detect_flats('inflated_dem')

# Plot flats
fig, ax = plt.subplots(figsize=(8,8))
plt.imshow(flats_test, cmap='cubehelix', zorder=1)
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.title('Flats', size=14)

#N    NE    E    SE    S    SW    W    NW
dirmap = (64,  128,  1,   2,    4,   8,    16,  32)

new_crs = CRS('epsg:4326')

# Compute flow direction based on corrected DEM
# grid.flowdir(data='inflated_dem', out_name='dir', dirmap=dirmap , as_crs=new_crs)
grid.flowdir(data='inflated_dem', out_name='dir', dirmap=dirmap)

plotFigure(grid.dir , 'flow dir', 'viridis')

# Compute flow accumulation based on computed flow direction
grid.accumulation(data='dir', out_name='acc', dirmap=dirmap)

accView = grid.view('acc', nodata=np.nan)
plotFigure(accView,"Cell Number",'PuRd')

###############################correr todo#####################################
# Delineate catchment at point of high accumulation
y, x = np.unravel_index(np.argsort(grid.acc.ravel())[-2], grid.acc.shape)
grid.catchment(x, y, data='dir', out_name='catch',
               dirmap=dirmap, xytype='index')

streams = grid.extract_river_network('catch', 'acc', threshold=200, dirmap=dirmap)########cambiar threshold
streams["features"][:2]

print(streams)

for new in streams['features']:
  line = np.asarray(new['geometry']['coordinates'])
  print(line)
  
 
fig, ax = plt.subplots(figsize=(6.5,6.5))

plt.grid('on', zorder=0)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Red Fluvial (>200 acumulación)')
plt.xlim(grid.bbox[0], grid.bbox[2])
plt.ylim(grid.bbox[1], grid.bbox[3])
ax.set_aspect('equal')

for stream in streams['features']:
    line = np.asarray(stream['geometry']['coordinates'])
    plt.plot(line[:, 0], line[:, 1])
    
#################################hhasta aqui#######################################
    
def saveDict(dic,file):
    f = open(file,'w')
    f.write(str(dic))
    f.close()


#save geojson as separate file
#saveDict(streams,'D:/T_JONA/TESIS_PISCO/srtm_90_cg/streams_200.geojson')

streamNet = gpd.read_file('D:/T_JONA/TESIS_PISCO/srtm_90_cg/streams_200.geojson')
# streamNet.crs = {'init' :'epsg:32613'}
streamNet.crs = CRS('epsg:32718')

# The polygonize argument defaults to the grid mask when no arguments are supplied
shapes = grid.polygonize()

# Plot catchment boundaries
fig, ax = plt.subplots(figsize=(6.5, 6.5))

for shape in shapes:
    coords = np.asarray(shape[0]['coordinates'][0])
    ax.plot(coords[:,0], coords[:,1], color='cyan')
    
ax.set_xlim(grid.bbox[0], grid.bbox[2])
ax.set_ylim(grid.bbox[1], grid.bbox[3])
ax.set_title('Límite de captación (vector)')
gpd.plotting.plot_dataframe(streamNet, None, cmap='winter_r', ax=ax)####cmap='blues'
plt.xlabel('Longitude')
plt.ylabel('Latitude')

########################
ax = streamNet.plot()

#mplleaflet.display(fig=ax.figure , epsg=4326)

# plot de acumulación y captación
fig, ax = plt.subplots(2, 2, figsize=(16, 16))
ax[0,0].imshow(np.where(~flats, grid.view('acc') + 1, np.nan), zorder=1, cmap='plasma')#CMAP:viridis, plasma, inferno, magma, cividis.
ax[0,1].imshow(np.where(~flats, grid.view('acc') + 1, np.nan), zorder=1, cmap='plasma',
               norm=colors.LogNorm(vmin=1, vmax=grid.acc.max()))
ax[1,0].imshow(np.where(grid.catch, grid.catch, np.nan), zorder=1, cmap='plasma')
ax[1,1].imshow(np.where(grid.catch, grid.view('acc') + 1, np.nan), zorder=1, cmap='plasma',
               norm=colors.LogNorm(vmin=1, vmax=grid.acc.max()))

ax[0,0].set_title('Acumulación (escala lineal)', size=14)
ax[0,1].set_title('Acumulación (escala logarítmica)', size=14)
ax[1,0].set_title('Captación más grande', size=14)
ax[1,1].set_title('Mayor acumulación de captación (escala logarítmica)', size=14)


for i in range(ax.size):
    ax.flat[i].set_yticklabels([])
    streams_new = grid.extract_river_network(grid.catch, grid.view('acc') + 1, threshold=5000, dirmap=dirmap)
streams_new["features"][:2]

   
#https://github.com/NithinGopakumar/Disease_spread_modelling_GIS/blob/90ad9bca985f3acdddfb07a08e6fc63cde599c09/Copy_of_Tutorial1%20(1).ipynb
