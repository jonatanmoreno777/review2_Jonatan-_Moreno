# -*- coding: utf-8 -*-
"""
Created on Fri May 27 22:01:00 2022

@author: ASUS
"""
#https://github.com/Fodark/geospatial-unitn-exam/blob/4e80d9c2980b331def25d7ff40a14cc43620f0c1/analysis.ipynb
import pandas as pd
import geopandas as gpd

import folium
from folium.plugins import HeatMap
from folium.plugins import MarkerCluster
from folium.plugins import BeautifyIcon
import contextily as ctx
from branca.element import Figure
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
import seaborn as sns
import rtree

import pyrosm
import osmnx as ox
from shapely.geometry import Point, LineString
import pyproj
from shapely.ops import transform
from owslib.wms import WebMapService

wgs84 = pyproj.CRS('EPSG:4326')
utm18s = pyproj.CRS('EPSG:32718')
projection_transform = pyproj.Transformer.from_crs(wgs84, utm18s, always_xy=False).transform

df = pd.read_csv("D:/Clase R en Hdrología/Nueva carpeta/Estac_Virtual.csv")
df = df[['id', 'name', 'neighbourhood', 'latitude', 'longitude', 'price']]
airbnb_4326 = gpd.GeoDataFrame(df, crs='EPSG:4326', geometry=gpd.points_from_xy(df.longitude, df.latitude))

#plot
ax = airbnb_4326.plot(color="red", figsize=(9, 9))
ctx.add_basemap(ax,crs=airbnb_4326.crs.to_string(), source=ctx.providers.OpenStreetMap.HOT)

sns.jointplot(x='longitude', y='latitude', color="xkcd:dusky blue", data=airbnb_4326)

neigh_bnb = airbnb_4326.groupby('neighbourhood').id.count().to_frame('number_bnb').reset_index().sort_values(['number_bnb', 'neighbourhood'], ascending=[False, True])
neigh_bnb

neighborhoods_4326 = gpd.read_file("neighbourhoods.geojson")

ax = neighborhoods_4326.plot(figsize=(9, 9), facecolor="cyan", alpha=0.3, edgecolor="black", linewidth=2)

texts = []
for p, label in zip(neighborhoods_4326.geometry.representative_point(), neighborhoods_4326.neighbourhood):
    texts.append(plt.text(p.x, p.y, label, fontsize = 8, color="black",fontweight='bold'))

ctx.add_basemap(ax,crs=airbnb_4326.crs.to_string(), source=ctx.providers.OpenStreetMap.HOT)