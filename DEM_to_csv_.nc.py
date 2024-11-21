# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 14:34:52 2021

@author: ASUS
"""

import richdem as rd
import rasterio
from rasterio import plot
import geopandas as gpd
from shapely.geometry import Point
from rasterio.plot import show_hist
import matplotlib.pyplot as plt
 
#import gdal######################python 3.8 ok¡¡¡¡¡¡¡¡¡¡¡
#import geospatialtools.terrain_tools as terrain_tools
###https://github.com/NCristea/NCristeaGEE/blob/5656a1b70b46adc2ce45e23108d24258a9ebfd41/delta_swe_processing_2016.ipynb
#https://github.com/collinsemmalise/Priestly_Taylor_Python_Implementation/blob/master/Priestley-Taylor-Condensed.ipynb
ras = rasterio.open('D:/T_JONA/TESIS_PISCO/srtm_90_cg/merge_srtm.tif')

# Veamos qué contiene la instancia, el .tif
dir(ras)

#crs????
ras.crs
#obtener el número de bandas del dataset rasterio
ras.count
#obtener los límites del dataset rasterio
ras.bounds
#obtener el ancho y la altura del dataset rasterio en píxeles
raster_width = ras.width
raster_height = ras.height
'width= {}, height={}'.format(raster_width,raster_height)
#plot rasterio
rasterio.plot.show(ras,cmap='bone')
#crear un objeto band1 del dataset rasterio
ras_band1 = ras.read(1).astype('float64')
#obtener el valor de píxel pasando fila y columna de band1
#no puede obtener coordenadas de longitud y latitud de una banda rasterio ya que es solo una matriz numérica
ras.xy(1,5)
#crear un objeto richdem usando rasterio band1
rs_richdem = rd.rdarray(ras_band1, no_data=-9999)
#plot richdem
rd.rdShow(rs_richdem, axes=False, cmap='bone', figsize=(9, 6))
#crear ráster de pendiente usando richdem
rs_rich_slope = rd.TerrainAttribute(rs_richdem, attrib='slope_percentage')

#plot slop

rd.rdShow(rs_rich_slope, axes=True, cmap='YlOrBr', figsize=(9, 6))#COLOR DE PLANO A EMPINADOS
#crear aspecto usando richdem
rs_rich_aspect = rd.TerrainAttribute(rs_richdem, attrib='aspect')
#plot aspecto
rd.rdShow(rs_rich_aspect, axes=False, cmap='jet', figsize=(9, 6))

#generar un diccionario de las siguientes listas: ¶
#1. objetos puntuales de geometría (geomerías bien formadas)
#2. Atributos: elevación, pendiente y orientación

pnt_data = {'elevation':[],'slope':[],'aspect':[],'geometry':[]}

width_step = 7
height_step = 200

for i in range(0,raster_width,width_step):
    if i % 47 == 0 and i != 0:
        for j in range(200,raster_height-height_step,height_step):
            pnt_data['elevation'].append(round(rs_richdem[i,j],2))
            pnt_data['slope'].append(round(rs_rich_slope[i,j],2))
            pnt_data['aspect'].append(round(rs_rich_aspect[i,j],2))
            pnt_data['geometry'].append(Point(ras.xy(i,j)))
            
#Create a geopandas dataframe from the dictionary
gp_df = gpd.GeoDataFrame(pnt_data, crs="EPSG:4326")
#exportar el datafrane de geopandas a geojson
gp_df.to_file('D:/T_JONA/TESIS_PISCO/srtm_90_cg/random_pts%%.geojson', driver="GeoJSON")
#exportar el datafrane de geopandas a shapefile
gp_df.to_file('D:/T_JONA/TESIS_PISCO/srtm_90_cg/random_pnts.shp', driver="ESRI Shapefile")
#exportar .csv
gp_df.to_csv('D:/T_JONA/TESIS_PISCO/srtm_90_cg/random_pnts.csv')
#convertir a .nc
import pandas as pd
pd.read_csv('D:/T_JONA/TESIS_PISCO/srtm_90_cg/random_pnts.csv').to_xarray().to_netcdf('D:/T_JONA/TESIS_PISCO/srtm_90_cg/random_pnts.nc')

#crear datos xarray desde pandas
import xarray as xr##############no install xaray prolemas con xarray
xr = xr.Dataset.from_dataframe(gp_df) ######################.nc???????????????????

#union de DEM:
#https://github.com/Automating-GIS-processes/site/blob/908a721f59c847f9f53aa7575bdb0bdde11627f7/source/notebooks/Raster/raster-mosaic.ipynb
#https://github.com/Automating-GIS-processes/site/blob/908a721f59c847f9f53aa7575bdb0bdde11627f7/source/notebooks/Raster/reading-raster.ipynb
#UNIR POLIGONOS Y PUNTOS
#https://github.com/Automating-GIS-processes/site/blob/908a721f59c847f9f53aa7575bdb0bdde11627f7/source/notebooks/L3/02_point-in-polygon.ipynb


