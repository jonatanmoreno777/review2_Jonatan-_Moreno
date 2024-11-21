# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 15:38:06 2022

@author: ASUS
"""

import geopandas as gpd 
import pandas as pd

import folium
import webbrowser
from ipywidgets import interact
import json
import requests
from folium import FeatureGroup, LayerControl, Map, Marker
import numpy as np
import sys
from folium.plugins import FloatImage
from folium.plugins import MiniMap
from folium import plugins
import json
from folium.plugins import Draw
from folium.plugins import MeasureControl
from folium.plugins import MousePosition
import os
from folium.plugins import HeatMap


geo_df = gpd.read_file("D:/Qswat_cachi/Jonatan_tesis/Watershed/Shapes/subs1.shp") #geo_df = gpd.read_file("D:/Qswat_cachi/Jonatan_tesis/Watershed/Shapes/riv1.shp")

Peru= gpd.read_file("D:/T_JONA/TESIS_PISCO/DEPARTAMENTOS.shp")
Peru.to_file("D:/Qswat_cachi/Jonatan_tesis/Scenarios/Simulacion_Python/Peru.geojson", driver ="GeoJSON")

print(geo_df.crs)
geo_df_ll = geo_df.to_crs(epsg=4326)
geo_df_ll.head()

dataDF = pd.read_excel('D:/Qswat_cachi/Jonatan_tesis/Scenarios/Simulacion_Python/py.xlsx') #pip install openpyxl

m = folium.Map(location=[-12.958, -74.063], zoom_start=7, tiles="Stamen Toner",name ="Subbasin")
borderStyle ={'color':'green', 'weight':2, 'fill':False}

folium.GeoJson("D:/Qswat_cachi/Jonatan_tesis/Scenarios/Simulacion_Python/Peru.geojson", name= "Peru",
               style_function=lambda x: borderStyle).add_to(m)


shapesLayer = folium.FeatureGroup(name="circles").add_to(m)

folium.Circle(location=[-13.453, -74.442],
              radius=5000,
              fill=True,
              tooltip="This is a tooltip text",
              popup=folium.Popup("""<h2>This is a popup</h2><br/>
              This is a <b>Bocatoma Chicllarazo"</b><br/>
              <img src="D:/descarga _esgf/Cachi_data/Chicllarazo.png" alt="Trulli" style="max-width:100%;max-height:100%">""", max_width=5000)
              ).add_to(shapesLayer)

cuencas = folium.FeatureGroup('Subcuencas').add_to(m)

for itr in range(len(dataDF)):
    latVal = dataDF.iloc[itr]['lat']
    longVal = dataDF.iloc[itr]['long']
    subcVal = dataDF.iloc[itr]['Subcuencas']
    parmStr = dataDF.iloc[itr]['Parámetros']
    qVal = dataDF.iloc[itr]['Q_mmaño']
    clr = 'blue' if parmStr.lower()=='PRECIP' else 'red'
    cRad = qVal
    # derive the circle pop up html content 
    popUpStr = 'Subcuencas- {0}<br>Parámetros - {1}<br>Q_mmaño - {2} $mm/año$'.format(
        subcVal, parmStr, qVal)
    
    folium.Circle(
        location=[latVal,longVal],
        popup=folium.Popup(popUpStr, min_width=100, max_width=700),
        radius=cRad,
        color=clr,
        weight=2,
        fill_opacity=0.1,
        fill=True).add_to(cuencas)
    
m.get_root().html.add_child(folium.Element('''
<div style= 'position:fixed; left :50px; bottom: 50px; height: 100px; width: 150px; border: 2px solid gray; z-index:900; font-size:large'>
&nbsp; Q_mmaño types <br>
&nbsp; <i class='fa fa-circle'  
                  style= 'color:red'></i> &nbsp; PRECIP<br>
&nbsp; <i class='fa fa-circle'  
                  style= 'color:blue'></i> &nbsp; ET<br>
</div>
'''))

m.choropleth(
    geo_data=geo_df_ll, name= "Subcuencas",
    data=geo_df_ll,
    columns=['Subbasin', 'Area'],
    key_on='feature.properties.Subbasin',
    legend_name='Area(sq.m)', 
    fill_color='BuPu',
    fill_opacity=0.4,
    #line_opacity=0.2,
    highlight=True)
 

folium.LayerControl().add_to(m)
m.add_child(MeasureControl())

#Agregar json #https://geojson.io/#map=13/-13.4171/-74.3237
data =os.path.join('D:/descarga _esgf/Cachi_data/plots_jpg', 'ejem.json')
folium.GeoJson(data, name='Cuchoquesera').add_to(m)

# crear un marcador propio
logoIcon =folium.features.CustomIcon('D:/descarga _esgf/Cachi_data/plots_jpg/IngAgricola.jpg',icon_size=(50,50))
folium.Marker([ -12.8975, -72.9575], popup='<strong>Location Five</strong>',
              icon=logoIcon).add_to(m)

draw = Draw(export=True)
draw.add_to(m)

plugins.Fullscreen(
    position="topright",
    title="Expand me",
    title_cancel="Exit me",
    force_separate_button=True,
).add_to(m)

plugins.LocateControl().add_to(m)

icon_plane = plugins.BeautifyIcon(
    icon="plane", border_color="#b3334f", text_color="#b3334f", icon_shape="triangle"
)

icon_number = plugins.BeautifyIcon(
    border_color="#00ABDC",
    text_color="#00ABDC",
    number=10,
    inner_icon_style="margin-top:0;",
)

f = folium.map.FeatureGroup(name= "Grupo")

m.add_child(folium.LatLngPopup())
m.add_child(f) 

minimap = MiniMap(toggle_display=True,tile_layer="Stamen Terrain", position="topleft",zoom_level_offset=-4) #width=400, height=100
m.add_child(minimap)
plugins.Geocoder().add_to(m)
plugins.ScrollZoomToggler().add_to(m)

m.save("D:/Qswat_cachi/Jonatan_tesis/Scenarios/Simulacion_Python/mymap3.html")
webbrowser.open("D:/Qswat_cachi/Jonatan_tesis/Scenarios/Simulacion_Python/mymap3.html")




