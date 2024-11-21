# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 17:47:58 2021

@author: ASUS
"""

import warnings
import pandas as pd
import geopandas as gpd

hoods = gpd.read_file("D:/T_JONA/TESIS_PISCO/DEPARTAMENTOS.shp")

hoods.info()

hoods.plot(figsize=(40, 20))

assembly = gpd.read_file("D:/T_JONA/TESIS_PISCO/cachi_wg84_R.shp")
assembly.info()
ax = assembly.plot(figsize=(40, 20))


#Calcule el área de cada uno en millas cuadradas

def get_sqmi(row):
    feet = row.geometry.area
    return feet / 27878400



assembly['area'] = assembly.to_crs({'epsg:32718'}).apply(get_sqmi, axis=1)
https://github.com/datadesk/geopandas-intersection-area-example/blob/1e38f93971408faba21dc1123f3aed4facca3cf9/geopandas-intersection-area-example.ipynb
https://github.com/datadesk/altair-election-maps-example/blob/master/notebook.ipynb

import altair as alt
