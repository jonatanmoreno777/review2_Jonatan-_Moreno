# -*- coding: utf-8 -*-
"""
Created on Sun May 22 11:27:17 2022

@author: ASUS
"""
#https://ncar.github.io/hydrology/projects/AIST
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import matplotlib

# Crear una conexión de base de datos
con = sqlite3.connect('D:/Qswat_cachi/Jonatan_tesis/Scenarios/Simulacion_Python/result_664_monthly.db3') #de scenarios txtlnOut
cursor = con.cursor()

# Check table names  ejemplo rch
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())


# Consultar columnas de la tabla RCH
def table_columns(db, table_name):
    curs = db.cursor()
    sql = "select * from %s where 1=0;" % table_name
    curs.execute(sql)
    return [d[0] for d in curs.description]


table_columns(con, 'rch')

# Lea las columnas especificadas de la tabla RCH en un marco de datos(Dataframe) de pandas
# SELECCIONE RCH, YR, MO, FLOW_OUTcms de rch
df = pd.read_sql_query("SELECT RCH, YR, MO, FLOW_OUTcms from rch", con)
df = df.set_index(['MO'])
print(df.head(11))

# Finalmente, no olvide cerrar la conexión db
con.close()

# Más datos de flujo de proceso con pandas
# Definición de cuartos personalizados
quarters = {1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM', 6: 'JJA',
            7: 'JJA', 8: 'JJA', 9: 'SON', 10: 'SON', 11: 'SON', 12: 'DJF'}

# Realice estadísticas estacionales para cada alcance
ssndf = df.groupby(['RCH', quarters])['FLOW_OUTcms'].mean()
ssndf.head(5)

ssndf = ssndf.reset_index()
ssndf.set_index(['RCH'])
ssndf.head(5)

ssndf = ssndf.rename(index=str, columns={"level_1": "Temporadas"})
ssndf.head(5)

pivoted = ssndf.pivot(index='RCH', columns='Temporadas', values='FLOW_OUTcms')
pivoted.head()

# PLOTS
# Plot size to 15" x 7"
matplotlib.rc('figure', figsize=(15, 7))
# Font size to 14
matplotlib.rc('font', size=14)
# Display top and right frame lines
matplotlib.rc('axes.spines', top=True, right=True)
# Remove grid lines
matplotlib.rc('axes', grid=False)
# Set backgound color to white
matplotlib.rc('axes', facecolor='white')

pivoted.plot(
    kind='bar', title='Seasonal Mean Discharge from 1983 to 2016 ($m^3/s$)')


#############Calcular los cambios de escorrentía media estacional############################

def read_rch(db_name):
    con = sqlite3.connect(db_name)
    cursor = con.cursor()

    df = pd.read_sql_query("SELECT RCH, YR, MO, FLOW_OUTcms from rch", con)
    df = df.set_index(['MO'])
    con.close()
    return df


def calculate_ssnmean(df):
    quarters = {1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM', 6: 'JJA',
                7: 'JJA', 8: 'JJA', 9: 'SON', 10: 'SON', 11: 'SON', 12: 'DJF'}
    ssndf = df.groupby(['RCH', quarters])['FLOW_OUTcms'].mean()
    ssndf = ssndf.reset_index()
    ssndf.set_index(['RCH'])
    ssndf = ssndf.rename(index=str, columns={"level_1": "Temporadas"})
    pivoted = ssndf.pivot(
        index='RCH', columns='Temporadas', values='FLOW_OUTcms')
    return pivoted


# Leer escorrentía de referencia
db_name = 'D:/Qswat_cachi/Jonatan_tesis/Scenarios/Simulacion_Python/result_664_monthly.db3'
df_bsl = read_rch(db_name)
df_bsl.head()

# Leer escorrentía en el futuro
db_name = 'D:/Qswat_cachi/Jonatan_tesis/Scenarios/Simulacion_Python/result_664_monthly.db3'
df_fut = read_rch(db_name)
df_fut.head()

# Calcular la escorrentía media estacional
pivoted_bsl = calculate_ssnmean(df_bsl)
pivoted_fut = calculate_ssnmean(df_fut)
print(pivoted_fut.head())
print(pivoted_bsl.head())

# Calcular cambios estacionales
pivoted_ch = (pivoted_fut - pivoted_bsl)/pivoted_bsl*100.0
pivoted_ch.head()

#plots
# Plot size to 15" x 7"
matplotlib.rc('figure', figsize = (15, 7))
# Font size to 14
matplotlib.rc('font', size = 14)
# Display top and right frame lines
matplotlib.rc('axes.spines', top = True, right = True)
# Remove grid lines
matplotlib.rc('axes', grid = False)
# Set backgound color to white
matplotlib.rc('axes', facecolor = 'white')

ax = pivoted_ch.plot(kind='bar',                    
             title='Seasonal Mean Runoff Change between Baseline and Future Periods')
ax.axhline(y=0, xmin=-1, xmax=1, color='k', lw=2)
ax.set_ylabel('Runoff Change (%)')


import geopandas as gpd 
import pandas as pd
import matplotlib.pyplot as plt
import regionmask 
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Rectangle # For rectangles
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox


geo_df = gpd.read_file("D:/Qswat_cachi/Jonatan_tesis/Watershed/Shapes/subs1.shp")
arroyos = gpd.read_file("D:/Qswat_cachi/Jonatan_tesis/Watershed/Shapes/riv1.shp") #geo_df = gpd.read_file("D:/Qswat_cachi/Jonatan_tesis/Watershed/Shapes/riv1.shp")

print(geo_df.crs)
geo_df_ll = geo_df.to_crs(epsg=4326)
eo_df_ll = arroyos.to_crs(epsg=4326)
geo_df_ll.head()
#geo_df_ll['Area'] = geo_df_ll['Area'].map(geo_df_ll.Area/100)

############################ Graficar cuenca CACHI########################
fig, ax = plt.subplots(figsize=(16,12))
geo_df_ll.plot(ax=ax, legend = True,column="Subbasin",cmap='nipy_spectral') #geo_df_ll.plot(ax=ax, legend = True,column="Subbasin",cmap='gist_rainbow')
#leg = plt.legend(facecolor='black', framealpha=0.8) #cmap='nipy_spectral'
ax.set_facecolor('#faf5e6')
#plt.axhline(y=-13.2, c='gray', ls=':')
plt.grid()# Agregar grillas  #ax.set_axis_off()
plt.text(0.01, 0.98 ,"Cuenca Cachi", fontsize=14, transform=ax.transAxes, verticalalignment='top', color='blue')
ax.text(-74.46, -13.56, "Cuenca Cachi", style="italic",bbox=dict(facecolor='wheat', alpha=1, boxstyle='round'))
plt.rcParams.update({'font.size': 22})
#ax.set_title('Subcuencas')
#ax.set(xlabel='$Longitud$', ylabel='$Latitud$')
ax.annotate('Bocatoma Chicllarazo...\nCanal de conducción : 10.8 $m^{3}\,s^{-1}$',
            xy=(-74.32,-13.45),#xy=(-74.4,-13.45)
            xytext=(-74.25,-13.4),#xytext=(-74.3,-13.35)
            arrowprops={'facecolor':'green','shrink':0.05},fontsize=16.5)
ax.add_patch(Rectangle((-74.475, -13.51), 0.152, 0.12,
                       edgecolor ="red", facecolor="none", lw=2))

#arroyos.plot(ax=ax, marker='o', color='cyan', markersize=3, linewidth = 0.2)

#Cree una nueva columna llamada coordenadas, que se basa en la columna "geometría"
geo_df_ll['coords'] = geo_df_ll['geometry'].apply(lambda x: x.representative_point().coords[:])
geo_df_ll['coords'] = [coords[0] for coords in geo_df_ll['coords']]

geo_df_ll.head()

#ax = geo_df_ll.plot(column='Area', cmap='viridis', categorical=True, figsize=(14, 7))
#ax.set_axis_off()
for idx, row in geo_df_ll.iterrows():    
    ax.annotate(s=row['Subbasin'], xy=row['coords'], fontsize=16, horizontalalignment='center')
    
#Agregar una imagen
Chicllarazo = mpimg.imread('D:/descarga _esgf/Cachi_data/Chicllarazo.png')
imgebox = OffsetImage(Chicllarazo, zoom=.05)
xy =[-74.6,-12.88]
ab = AnnotationBbox(imgebox,xy,xybox = (1,1), boxcoords ="offset points")
ax.add_artist(ab)

Chicllarazo = mpimg.imread('D:/descarga _esgf/Cachi_data/chicllarazo_ok.png')
imgebox = OffsetImage(Chicllarazo, zoom=.19)
xy =[-74.4,-13.45]
abc = AnnotationBbox(imgebox,xy,xybox = (1,1), boxcoords ="offset points")
ax.add_artist(abc)

### Agregar un shapefile
eo_df_ll.plot(ax=ax, marker='o', color='cyan', markersize=3, linewidth = 0.6)

###desea agregar campo area
geo_df_ll.plot(ax=ax,
               column='Area', 
               scheme='fisher_jenks', 
               k=5, 
               cmap=plt.cm.nipy_spectral, 
               legend=True,
               linewidth = .05)
############################################################################
    
#Clasificar cuencas hidrográficas según el área usando PySAL
import pysal

f, ax = plt.subplots(1, figsize = (14, 7))
ax.set_title('Watersheds by area ($m^2$)')
geo_df_ll.plot(ax=ax,
               column='Area', 
               scheme='fisher_jenks', 
               k=7, 
               #cmap=plt.cm.Blues, 
               legend=True,
               linewidth = 0.5)
ax.set_axis_off()
for idx, row in geo_df_ll.iterrows():    
    plt.annotate(s=row['Subbasin'], xy=row['coords'], fontsize=16, horizontalalignment='center')
plt.axis('equal')


#https://docs.bokeh.org/en/latest/docs/reference/sampledata.html#sampledata-stocks   genial bokeh
#https://docs.bokeh.org/en/latest/docs/gallery/stocks.html
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

sys.path.insert(0, 'folium')
sys.path.insert(0, 'branca')

import branca
n = 100
lats = np.random.uniform(-12.9550, -13.3422, n)
lngs = np.random.uniform(-74.3836, -74.5972, n)
sizes = np.random.uniform(2, 20, n)
colors = np.random.uniform(0, 50, n)
cm = branca.colormap.LinearColormap(['green', 'yellow', 'red'], vmin=0, vmax=50)
print(cm(25))
cm
f = folium.map.FeatureGroup(name= "Grupo")

for lat, lng, size, color in zip(lats, lngs, sizes, colors):
    f.add_child(
        folium.features.CircleMarker(
            [lat, lng],
            radius=size,
            color=None,
            fill_color=cm(color),
            fill_opacity=0.6)
        )
m = folium.Map([-12.958, -74.063], zoom_start=5, tiles='Stamen Toner')

url = (
    "https://raw.githubusercontent.com/python-visualization/folium/main/examples/data"
)

vis1 = json.loads(requests.get(f"{url}/vis1.json").text)
vis2 = json.loads(requests.get(f"{url}/vis2.json").text)
vis3 = json.loads(requests.get(f"{url}/vis3.json").text)


Peru= gpd.read_file("D:/T_JONA/TESIS_PISCO/DEPARTAMENTOS.shp")
Peru.to_file("D:/Qswat_cachi/Jonatan_tesis/Scenarios/Simulacion_Python/Peru.geojson", driver ="GeoJSON")
m = folium.Map(location=[-12.958, -74.063], zoom_start=7, tiles="Stamen Toner",name ="Subbasin")#tiles="Stamen Terrain"
folium.GeoJson("D:/Qswat_cachi/Jonatan_tesis/Scenarios/Simulacion_Python/red_arroyos.geojson", name= "red_arroyos").add_to(m)
#folium.GeoJson("D:/Qswat_cachi/Jonatan_tesis/Scenarios/Simulacion_Python/Peru.geojson", name= "Peru").add_to(m)

datos = [[-12.9216,  -74.6631, 400],
         [-13.7154,   -74.0643, 500],
         [-13.9767,  -75.1108, 800],
         ]

HeatMap(datos, name= 'prec').add_to(m) #https://www.youtube.com/watch?v=LV1ykkQkx40

#ls = folium.PolyLine(locations=[[-13.7421],[-75.0806],[-13.7074],[-75.0806],[-13.7421]],color='blue')
#ls.add_to(m)

m.add_child(folium.LatLngPopup())
m.add_child(f) 

minimap = MiniMap(toggle_display=True,tile_layer="Stamen Terrain", position="topleft",zoom_level_offset=-4) #width=400, height=100
m.add_child(minimap)
plugins.Geocoder().add_to(m)
plugins.ScrollZoomToggler().add_to(m)
#plugins.Terminator().add_to(m)
#plugins.BoatMarker(location=(-12.9550, -74.2504), heading=45, wind_heading=150, wind_speed=45, color="#8f8").add_to(m)

icon_plane = plugins.BeautifyIcon(
    icon="plane", border_color="#b3334f", text_color="#b3334f", icon_shape="triangle"
)

icon_number = plugins.BeautifyIcon(
    border_color="#00ABDC",
    text_color="#00ABDC",
    number=10,
    inner_icon_style="margin-top:0;",
)
folium.Marker(location=[-13.2159, -74.0472], popup="Portland, OR", icon=icon_plane).add_to(m)
folium.Marker(location=[-13.2279,  -74.5189], popup="Portland, OR", icon=icon_number).add_to(m)

plugins.Fullscreen(
    position="topright",
    title="Expand me",
    title_cancel="Exit me",
    force_separate_button=True,
).add_to(m)


plugins.LocateControl().add_to(m)

html = """
    <h1> Ventana emergente</h1><br>
    lineas de codigo...
    <p>
    <code>
        from numpy import *<br>
        exp(-2*pi)
    </code>
    </p>
    """
iframe = branca.element.IFrame(html=html, width=500, height=300)
popup = folium.Popup(iframe, max_width=500)
folium.Marker([-13.1611, -74.2923], popup=html).add_to(m)


##################################################
draw = Draw(export=True)
draw.add_to(m)

# crear un marcador propio
logoIcon =folium.features.CustomIcon('D:/descarga _esgf/Cachi_data/plots_jpg/IngAgricola.jpg',icon_size=(50,50))
folium.Marker([ -12.8975, -72.9575], popup='<strong>Location Five</strong>',
              icon=logoIcon).add_to(m)

#Agregar json #https://geojson.io/#map=13/-13.4171/-74.3237
data =os.path.join('D:/descarga _esgf/Cachi_data/plots_jpg', 'ejem.json')
folium.GeoJson(data, name='Cuchoquesera').add_to(m)
##############################
m.add_child(MeasureControl())
#MousePosition().add_to(m)
formatter = "function(num) {return L.Util.formatNum(num, 3) + ' º ';};"

MousePosition(
    position="topright",
    separator=" | ",
    empty_string="NaN",
    lng_first=True,
    num_digits=20,
    prefix="Coordinates:",
    lat_formatter=formatter,
    lng_formatter=formatter,
).add_to(m)

###################################################
folium.Marker(
    location=[-12.9550, -74.2504],
    popup=folium.Popup(max_width=450).add_child(
        folium.Vega(vis1, width=450, height=250)
    ),
).add_to(m)

folium.Marker(
    location=[-13.0541, -74.190],
    popup=folium.Popup(max_width=450).add_child(
        folium.Vega(vis2, width=450, height=250)
    ),
).add_to(m)

folium.Marker(
    location=[-13.2159, -74.0472],
    popup=folium.Popup(max_width=450).add_child(
        folium.Vega(vis3, width=450, height=250)
    ),
).add_to(m)

####################################################
# Si desea obtener la posición del dispositivo de usuario después de cargar el mapa, establezca
#plugins.LocateControl(auto_start=True).add_to(m)

folium.feature_group = FeatureGroup(name="Subbasins")
folium.Marker(location=[-12.9550, -74.2504], popup="M.C Huanta").add_to(folium.feature_group)
folium.Marker(location=[-13.0541, -74.190], popup="M.C Pongora").add_to(folium.feature_group)
folium.Marker(location=[-13.2159, -74.0472], popup="M.C Yucaes").add_to(folium.feature_group)
folium.Marker(location=[-13.2286, -74.2017], popup="M.C Huatatas").add_to(folium.feature_group)
folium.Marker(location=[-13.3756, -74.2793], popup="M.C Chicllarazo (aguas abajo)").add_to(folium.feature_group)
folium.Marker(location=[-13.4444, -74.4070], popup="M.C Chicllarazo (aguas abajo)").add_to(folium.feature_group)
folium.Marker(location=[-13.3422, -74.5972], popup="M.C Apacheta").add_to(folium.feature_group)
folium.Marker(location=[-13.2279,  -74.5189], popup="M.C Paccha").add_to(folium.feature_group)
folium.Marker(location=[-13.2667, -74.3809], popup="M.C Vinchos").add_to(folium.feature_group)
folium.Marker(location=[-13.0641, -74.3836], popup="M.C Huanta").add_to(folium.feature_group)
folium.Marker(location=[-13.1611, -74.2923], popup="M.C Chillico").add_to(folium.feature_group)

folium.feature_group.add_to(m)

#m.add_child(folium.ClickForMarker(popup="Waypoint"))
folium.Marker(
    location=[-13.453, -74.442],
    popup="Bocatoma Chicllarazo", tooltip='M = 4.0',
    icon=folium.Icon(color="green", icon= 'leaf'),
).add_to(m)
folium.Marker(
    location=[-12.901, -74.323],
    popup="Cachi",
    icon=folium.Icon(icon="cloud"),
).add_to(m)

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

folium.Circle(
    radius=100,
    location=[-13.453, -74.442],
    popup="The Waterfront",
    color="crimson",
    fill=False,
).add_to(m)

folium.CircleMarker(
    location=[-12.901, -74.323],
    radius=50,
    popup="Laurelhurst Park",
    color="#3186cc",
    fill=True,
    fill_color="#3186cc",
).add_to(m)

folium.TileLayer('openstreetmap').add_to(m)
folium.TileLayer('stamenterrain', attr='stamenterrain').add_to(m)
folium.TileLayer('stamentwatercolor', attr='stamentwatercolor').add_to(m)
folium.TileLayer('CartoDBpositron', attr='CartoDBpositron').add_to(m)
folium.TileLayer('CartoDBdark_matter', attr='CartoDBdark_matter').add_to(m)

folium.LayerControl().add_to(m)#folium.LayerControl(collapsed=False).add_to(m)

m



m.save("D:/Qswat_cachi/Jonatan_tesis/Scenarios/Simulacion_Python/mymap3.html")
webbrowser.open("D:/Qswat_cachi/Jonatan_tesis/Scenarios/Simulacion_Python/mymap3.html")
folium.GeoJs
#https://python-visualization.github.io/folium/quickstart.html#GeoJSON/TopoJSON-Overlays

###############################3
#https://nbviewer.org/github/python-visualization/folium_contrib/tree/master/notebooks/
tiles = [name.strip() for name in """
    OpenStreetMap
    Mapbox Bright
    Mapbox Control Room
    Stamen Terrain
    Stamen Toner
    Stamen Watercolor
    CartoDB positron
    CartoDB dark_matter""".strip().split('\n')]

@interact(lat=(-90, 90), lon=(-180, 180), tiles=tiles, zoom=(1, 18))
def create_map(lat=-12.958, lon=-74.063, tiles="Stamen Toner", zoom=10):
    return folium.Map(location=(lat, lon), tiles=tiles, zoom_start=zoom)


#https://github.com/python-visualization/folium/tree/main/examples ---------Seguir con Folium
#https://github.com/python-visualization/folium/blob/main/examples/plugin-Search.ipynb




#https://github.com/royalosyin/pySQLiteSWAT/blob/f07f95763fe89ef0ba1ceb730ec2b35a66db0879/ex4-Postprocess%20SWAT%20Simulations%20(4)%20-%20Water%20Yield%20Changes%20on%20Map.ipynb

lines = [
    {
        "coordinates": [
            [-13.439771, -74.307394],
            [-13.421506, -74.332433],
        ],
        "dates": ["2017-06-02T00:00:00", "2017-06-02T00:10:00"],
        "color": "red",
    },
    {
        "coordinates": [
            [-13.421506, -74.332433],
            [-13.439992, -74.352140],
        ],
        "dates": ["2017-06-02T00:10:00", "2017-06-02T00:20:00"],
        "color": "blue",
    },
    {
        "coordinates": [
            [-13.439992, -74.352140],
            [-13.456394, -74.326972],
        ],
        "dates": ["2017-06-02T00:20:00", "2017-06-02T00:30:00"],
        "color": "green",
        "weight": 15,
    },
    {
        "coordinates": [
            [-13.456394, -74.326972],
            [-13.439771, -74.307394],
        ],
        "dates": ["2017-06-02T00:30:00", "2017-06-02T00:40:00"],
        "color": "#FFFFFF",
    },
]

features = [
    {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": line["coordinates"],
        },
        "properties": {
            "times": line["dates"],
            "style": {
                "color": line["color"],
                "weight": line["weight"] if "weight" in line else 5,
            },
        },
    }
    for line in lines
]

plugins.TimestampedGeoJson(
    {
        "type": "FeatureCollection",
        "features": features,
    },
    period="PT1M",
    add_last_point=True,
).add_to(m)

import random

import folium
from scipy.spatial import ConvexHull


# Function to create a list of some random points
def randome_points(amount, LON_min, LON_max, LAT_min, LAT_max):

    points = []
    for _ in range(amount):
        points.append(
            (random.uniform(LON_min, LON_max), random.uniform(LAT_min, LAT_max))
        )

    return points


# Function to draw points in the map
def draw_points(map_object, list_of_points, layer_name, line_color, fill_color, text):

    fg = folium.FeatureGroup(name=layer_name)

    for point in list_of_points:
        fg.add_child(
            folium.CircleMarker(
                point,
                radius=1,
                color=line_color,
                fill_color=fill_color,
                popup=(folium.Popup(text)),
            )
        )

    map_object.add_child(fg)

def create_convexhull_polygon(
    map_object, list_of_points, layer_name, line_color, fill_color, weight, text
):

    # Since it is pointless to draw a convex hull polygon around less than 3 points check len of input
    if len(list_of_points) < 3:
        return

    # Create the convex hull using scipy.spatial
    form = [list_of_points[i] for i in ConvexHull(list_of_points).vertices]

    # Create feature group, add the polygon and add the feature group to the map
    fg = folium.FeatureGroup(name=layer_name)
    fg.add_child(
        folium.vector_layers.Polygon(
            locations=form,
            color=line_color,
            fill_color=fill_color,
            weight=weight,
            popup=(folium.Popup(text)),
        )
    )
    map_object.add_child(fg)

    return map_object

list_of_points = randome_points(
    amount=10, LON_min=48, LON_max=49, LAT_min=9, LAT_max=10
)

create_convexhull_polygon(
    m,
    list_of_points,
    layer_name="Example convex hull",
    line_color="lightblue",
    fill_color="lightskyblue",
    weight=5,
    text="Example convex hull",
)

draw_points(
    m,
    list_of_points,
    layer_name="Example points for convex hull",
    line_color="royalblue",
    fill_color="royalblue",
    text="Example point for convex hull",
)