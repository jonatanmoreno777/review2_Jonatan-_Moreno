# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 21:49:48 2021

@author: ASUS
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from pyproj import CRS
import matplotlib.colors as colors

gdf_koppen = gpd.read_file("D:/T_JONA/TESIS_PISCO/DEPARTAMENTOS.shp")
gdf_koppen



# Check
gdf_koppen.plot(facecolor='none',edgecolor='k')
ax = plt.gca()
gdf_koppen.plot(ax=ax,facecolor='none',edgecolor='b');
#visualizar coordenadas
gdf_koppen.crs
print(gdf_koppen.crs)
#convertir coordenadas
gdf_koppen.to_crs({'init':'epsg:32718'}, inplace=True)
# otra opcion convertir de wgs84 a utm:
gdf_koppen.crs = CRS.from_epsg(32718).to_wkt() 
print(gdf_koppen.crs)
#gdf_koppen.to_file('D:/T_JONA/TESIS_PISCO/gdf_koppen.json', driver='GeoJSON')


##############################plot
# Inspect file
gdf_koppen

cmap = colors.ListedColormap(['darkred', 'red', 'darksalmon', 'mistyrose',
                            'yellow', 'orange', 'burlywood', 'chocolate',
                            'darkgreen', 'darkolivegreen', 'green', 'lawngreen', 'limegreen', 'greenyellow', 'peru', 'darkgoldenrod', 'saddlebrown',
                            'darkmagenta', 'darkviolet', 'magenta', 'crimson', 'violet', 'pink', 'mediumpurple', 'slategrey', 'slateblue', 'rebeccapurple', 'darkorchid', 'orchid',
                            'steelblue', 'blue',
                            'aqua'])

boundaries = list(range(1,33))

fig, ax = plt.subplots(figsize=(30, 20))
gdf_koppen.plot(ax=ax, column='DEPARTAMEN', legend=True, cmap=cmap)

plt.rcParams.update({'font.size': 22})
ax.set_title('Koppen-Geiger Climate Zones')
#plt.savefig('Koppen_Geiger.png')

#https://github.com/jeromaerts/teaching_supervision_code/blob/main/sophie_bsc/Plot_Koppen_Geiger_Sophie.ipynb
#https://github.com/cheginit/HyRiver-examples/tree/main/notebooks#############gis python
#https://github.com/emiliom/geopandas-tutorial-maptime/tree/master/notebooks###############geopandas 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pysheds.grid import Grid
import seaborn as sns
import warnings

import rasterio
import mplleaflet
import geopandas as gpd

grid = Grid.from_raster('D:/C/CUENCA _SWAT/topografiadem_cachi_srtm/srtm_22_15.tif', data_name='dem')


def plotFigure(data, label, cmap='Blues'):
    plt.figure(figsize=(12,10))
    plt.imshow(data ,extent=grid.extent, cmap = cmap)
    plt.colorbar(label=label)
    plt.grid()

# # Plot the raw DEM data
fig, ax = plt.subplots(figsize=(12,10))
plt.imshow(grid.dem, extent=grid.extent, cmap='terrain', zorder=1)#cmap='cubehelix'
plt.colorbar(label='Elevation (m)')
plt.title('Digital elevation map')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# elevDem=grid.dem[:-1,:-1]
plotFigure(grid.dem, 'Elevation (m)')

dirmap = (64,  128,  1,   2,    4,   8,    16,  32)

grid.flowdir(data='dem', out_name='dir' , dirmap=dirmap)

plotFigure(grid.dir , 'Flowdirection' , 'viridis')

# Specify pour point
# x, y = -74.688, -13.582
x, y = -74.324, -12.901##################aforo_cachi

# Delineate the catchment
grid.catchment(data='dir', x=x, y=y, dirmap=dirmap, out_name='catch',
               recursionlimit=15000, xytype='label', nodata_out=0)

#Sujete el cuadro delimitador a la cuenca de captación
grid.clip_to('catch')

demview = grid.view('dem', nodata=np.nan)

plotFigure(demview , "elevation(m)")

grid.to_raster(demview , 'demview.tif')

#Obtener acumulación de flujo
grid.accumulation(data='catch', dirmap=dirmap ,pad_inplace=False, out_name='acc')

accView = grid.view('acc', nodata=np.nan)
plotFigure(accView,"Cell Number",'PuRd')

streams = grid.extract_river_network('catch','acc',threshold=200 , dirmap=dirmap)
# streams["features"][:2]
streams

def savedict(dic,file):
    f = open(file,'w')
    f.write(str(dic))
    f.close()
    
savedict(streams , 'D:/C/CUENCA _SWAT/topografiadem_cachi_srtm/streams.geojson')   

streamnet = gpd.read_file('D:/C/CUENCA _SWAT/topografiadem_cachi_srtm/streams.geojson')
streamnet.crs = {'init' :'epsg:4326'}    



# The polygonize argument defaults to the grid mask when no arguments are supplied
shapes = grid.polygonize()

# Plot catchment boundaries
fig, ax = plt.subplots(figsize=(6.5, 6.5))

for shape in shapes: 
    coords = np.asarray(shape[0]['coordinates'][0]) 
    ax.plot(coords[:,0], coords[:,1], color='cyan')
    
ax.set_xlim(grid.bbox[0], grid.bbox[2])
ax.set_ylim(grid.bbox[1], grid.bbox[3])
ax.set_title('Catchment boundary (vector)')
gpd.plotting.plot_dataframe(streamnet, None, cmap='Blues', ax=ax)


#ax = streamnet.plot()
#mplleaflet.display(fig=ax.figure, crs=streamnet.crs, tiles='esri_aerial')    
    
 #############################3Crear una cuadrícula de dirección de flujo a partir de un ráster   
#https://github.com/NithinGopakumar/Disease_spread_modelling_GIS/blob/90ad9bca985f3acdddfb07a08e6fc63cde599c09/extract_river_network.ipynb     

grid = Grid.from_raster('D:/C/CUENCA _SWAT/topografiadem_cachi_srtm/srtm_22_15.tif', data_name='dem') 

#Especificar valores de dirección de flujo 
#N    NE    E    SE    S    SW    W    NW
dirmap = (64,  128,  1,   2,    4,   8,    16,  32) 

#Delimitar captación 
# Specify pour point
x, y = -74.324, -12.901

# Delimitar captación
grid.catchment(data='dem', x=x, y=y, dirmap=dirmap, out_name='catch',
               recursionlimit=15000, xytype='label')

# Sujete el cuadro delimitador a la cuenca
grid.clip_to('catch')   

#Obtener acumulación de flujo    
grid.accumulation(data='catch', dirmap=dirmap, pad_inplace=False, out_name='acc')

#Extraer la red fluvial    
branches = grid.extract_river_network('catch', 'acc', threshold=200, dirmap=dirmap)

#plot river
fig, ax = plt.subplots(figsize=(12,10))

plt.grid('on', zorder=0)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('River network (>200 accumulation)')#################ojo lat y long falla
plt.xlim(grid.bbox[0], grid.bbox[2])
plt.ylim(grid.bbox[1], grid.bbox[3])
ax.set_aspect('equal')

for branch in branches['features']:
    line = np.asarray(branch['geometry']['coordinates'])
    plt.plot(line[:, 0], line[:, 1])    
    
    
branches = grid.extract_river_network('catch', 'acc', threshold=50, dirmap=dirmap)

fig, ax = plt.subplots(figsize=(6.5,6.5))

plt.grid('on', zorder=0)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('River network (>50 accumulation)')
plt.xlim(grid.bbox[0], grid.bbox[2])
plt.ylim(grid.bbox[1], grid.bbox[3])
ax.set_aspect('equal')

for branch in branches['features']:
    line = np.asarray(branch['geometry']['coordinates'])
    plt.plot(line[:, 0], line[:, 1])

#plt.savefig('img/river_network.png', bbox_inches='tight')

branches = grid.extract_river_network('catch', 'acc', threshold=2, dirmap=dirmap)

fig, ax = plt.subplots(figsize=(6.5,6.5))

plt.grid('on', zorder=0)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('River network (>2 accumulation)')
plt.xlim(grid.bbox[0], grid.bbox[2])
plt.ylim(grid.bbox[1], grid.bbox[3])
ax.set_aspect('equal')

for branch in branches['features']:
    line = np.asarray(branch['geometry']['coordinates'])
    plt.plot(line[:, 0], line[:, 1])
    
    
####################################Instalar una cuadrícula desde un ráster DEM

grid = Grid.from_raster('D:/C/CUENCA _SWAT/topografiadem_cachi_srtm/srtm_22_15.tif', data_name='dem') 

#Definir una función para trazar el modelo de elevación digital

def plotFigure(data, label, cmap='Purples'):
    plt.figure(figsize=(12,10))
    plt.imshow(data ,extent=grid.extent, cmap = 'viridis')
    plt.colorbar(label=label, cmap=cmap)
    plt.grid()
    
# fig, ax = plt.subplots(figsize=(8,6))
# fig.patch.set_alpha(0)

# plt.imshow(grid.dem, extent=grid.extent, cmap='cubehelix', zorder=1)
# plt.colorbar(label='Elevation (m)')
# plt.grid(zorder=0)
# plt.title('Digital elevation map')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.tight_layout()
# plt.savefig('conditioned_dem.png', bbox_inches='tight')

#Minnor cortando bordes para realzar colobars

elevDem=grid.dem[:-1,:-1]
plotFigure(elevDem, 'Elevation (m)')    

derpressions = grid.detect_depressions('dem')

plt.imshow(derpressions)

grid.fill_depressions(data= 'dem', out_name= 'flooded_dem')

derpressions = grid.detect_depressions('flooded_dem')
plt.imshow(derpressions)

#Resolver pisos en DEM
flats = grid.detect_flats('flooded_dem')
plt.imshow(flats)

grid.resolve_flats(data='flooded_dem', out_name='inflated_dem')
plt.imshow(grid.inflated_dem[:-1,:-1])

#direccion de flujo
#N    NE    E    SE    S    SW    W    NW
dirmap = (64,  128,  1,   2,    4,   8,    16,  32)

#Convertir DEM en cuadrícula de dirección de flujo
grid.flowdir(data='inflated_dem', out_name='dir', dirmap=dirmap)

plotFigure(grid.dir,'Flow Direction','Greens')

#plot flow direccion
#fig = plt.figure(figsize=(8,6))
#fig.patch.set_alpha(0)

#plt.imshow(grid.dir, extent=grid.extent, cmap='viridis', zorder=2)
#boundaries = ([0] + sorted(list(dirmap)))
#plt.colorbar(boundaries= boundaries,
              #values=sorted(dirmap))
#plt.xlabel('Longitude')
#plt.ylabel('Latitude')
#plt.title('Flow direction grid')
#plt.grid(zorder=-1)
#plt.tight_layout()

#plt.savefig('D:/C/CUENCA _SWAT/topografiadem_cachi_srtm/flow_direction.png', bbox_inches='tight')

#Leer una cuadrícula de dirección de flujo desde un ráster

streamnet = gpd.read_file('D:/C/CUENCA _SWAT/topografiadem_cachi_srtm/streams.geojson')
streamnet.crs = {'init' :'epsg:4326'}  

# The polygonize argument defaults to the grid mask when no arguments are supplied
shapes = grid.polygonize()
# Plot catchment boundaries
fig, ax = plt.subplots(figsize=(6.5, 6.5))
for shape in shapes:
    coords = np.asarray(shape[0]['coordinates'][0])
    ax.plot(coords[:,0], coords[:,1], color='cyan')

ax.set_xlim(grid.bbox[0], grid.bbox[2])
ax.set_ylim(grid.bbox[1], grid.bbox[3])
ax.set_title('Catchment boundary (vector)')
gpd.plotting.plot_dataframe(streamnet, None, cmap='Blues', ax=ax)
#ax = streamNet.plot()  

# The polygonize argument defaults to the grid mask when no arguments are supplied
shapes = grid.polygonize()

# Plot catchment boundaries
fig, ax = plt.subplots(figsize=(6.5, 6.5))

for shape in shapes: 
    coords = np.asarray(shape[0]['coordinates'][0]) 
    ax.plot(coords[:,0], coords[:,1], color='cyan')
    
ax.set_xlim(grid.bbox[0], grid.bbox[2])
ax.set_ylim(grid.bbox[1], grid.bbox[3])
ax.set_title('Catchment boundary (vector)')
gpd.plotting.plot_dataframe(streamnet, None, cmap='Blues', ax=ax)

#########continuara¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡https://github.com/NithinGopakumar/Disease_spread_modelling_GIS/blob/90ad9bca985f3acdddfb07a08e6fc63cde599c09/Copy%20of%20quickstart.ipynb


















