# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 11:44:34 2022

@author: ASUS
"""

#*****************************************************************************************#
#           ANALISIS DEL NUEVO PRODUCTO GRILLADO PARA PERÚ Y ECUADOR (RAIN4PE) 
#                           USANDO PYTHON PARA LA CUENCA CACHI                 

#*****************************************************************************************#
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import cartopy.feature as cfeature

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']},size=12)
rc('text', usetex=False)

# Data diaria
ds = xr.open_dataset("D:/R/RAIN4PE_daily_0.1d_1981_2015_v1.0.nc")

seasonal_mean=ds.groupby('time.month').mean()
seasonal_mean=seasonal_mean.reindex(month=[1,2,3,4,5,6,7,8,9,10,11,12])
seasonal_mean.pcp.plot(col='month', robust= True, cmap='hsv',col_wrap=4)

ds.groupby('time.season')
seasonal_mean=ds.groupby('time.season').mean()
seasonal_mean=seasonal_mean.reindex(season=['DJF', 'JJA' ,'MAM' ,'SON'])
g= seasonal_mean.pcp.plot(col='season', robust= True, cmap='jet',col_wrap=4)

bottomright = g.axes[-1, -1]
bottomright.annotate("RAIN4PE", (-75, -8), color='red')

#*****************************************************************************************#
#                    COMPARACIÓN (TEMPORADAS) PISCO_SENAMHI - RAIN4PE                 
#*****************************************************************************************#
piscoday = xr.open_dataset("D:/T_JONA/TESIS_PISCO/Entrada/Pisco_Pp/Precday.nc")
RAINday = xr.open_dataset("D:/R/RAIN4PE_daily_0.1d_1981_2015_v1.0.nc")

print(piscoday.longitude.attrs)
seas_pp = piscoday.groupby('z.season').mean()
seas_pcp = RAINday.groupby('time.season').mean()

from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

hp_shp = Reader("D:/T_JONA/TESIS_PISCO/DEPARTAMENTOS.shp")
print(seas_pp, seas_pcp)
## Obtener extensión de coordenadas
long_day = seas_pp['variable'].longitude.values 
lat_day = seas_pp['variable'].latitude.values 

long_pcp = seas_pcp['pcp'].Longitude.values 
lat_pcp = seas_pcp['pcp'].Latitude.values

import matplotlib
# Trazar los plots..
matplotlib.rcParams['font.size'] =12
plt.rcParams["axes.linewidth"]  = 1.5
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = 'New Century Schoolbook'
plt.rcParams["xtick.major.size"]=9
plt.rcParams["ytick.major.size"]=9
plt.rcParams['ytick.right'] =True
plt.rcParams['xtick.top'] =True

# Definir etiquetas de los ejes 'x' y 'y'
x_tick_labels = [u'84\N{DEGREE SIGN}W',u'80 \N{DEGREE SIGN}W',
                 u'75\N{DEGREE SIGN}W',u'70\N{DEGREE SIGN}W',
                 u'65\N{DEGREE SIGN}W']

y_tick_labels = [u'20\N{DEGREE SIGN}S',u'16\N{DEGREE SIGN}S',u'12 \N{DEGREE SIGN}S',
                 u'8 \N{DEGREE SIGN}S',u'4 \N{DEGREE SIGN}S',
                 u'0\N{DEGREE SIGN}N',u'4\N{DEGREE SIGN}N']

# Map con cartopy
projection = ccrs.PlateCarree()
## Establecer el tamaño y dimension de los plots...
fig, axes = plt.subplots(2,4,sharex=True, sharey=True,subplot_kw=dict(projection=ccrs.PlateCarree()))
fig.set_size_inches(18,10)

# Trazar datos de temporadas
for i, season in enumerate(('DJF','MAM','JJA', 'SON')):
    
    o= axes[0,i].contourf(long_day, lat_day,seas_pp['variable'].sel(season=season),
                          transform=projection,cmap='hsv',hatches=['/'])
    p= axes[1,i].contourf(long_pcp,lat_pcp,seas_pcp['pcp'].sel(season=season),
                          transform=projection,cmap='hsv',hatches=['/'])
    
    axes[0, i].set_title(season,fontsize=16,fontweight='bold')

# agregue líneas costeras y fronteras de países a la gráfica de contorno
for ax in axes.flat:
    ax.set_xlim(-84,-65)
    ax.set_ylim(-20,4)
    ax.set_xticks(np.linspace(-84,-65,5), crs=projection)
    ax.set_yticks(np.linspace(-20,4,4), crs=projection)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidths=1)
    ax.axes.axis('tight')
    ax.add_feature(ShapelyFeature(hp_shp.geometries(),ccrs.PlateCarree(), 
                                  edgecolor='black'), facecolor='none')
    ax.add_feature(cfeature.RIVERS)
    
# Oculte las etiquetas 'x' y marque las etiquetas de los plots...
for ax in axes.flat:
    ax.label_outer()
    ax.set_xticks([-84,-80,-75,-70,-65])
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticks([-20,-16,-12,-8,-4,0,4])
    ax.set_yticklabels(y_tick_labels)
    
# Asignar etiqueta al eje y
for i,rowlabel in enumerate(('PISCOpp' ,'RAIN4PE')):
    axes[i, 0].set_ylabel(rowlabel,fontsize=16,fontweight='bold')

for ax in axes[-1,:].flatten():
    ax.set_xlabel('Longitud',fontsize=16,fontweight='bold')

# Aplanar ejes y obtener posiciones de los plots...
DP = axes.T.flatten()

# Ajuste la barra de colores al lado derecho de cada fila
# donde arg es [izquierda, abajo, ancho, alto]
cax1=fig.add_axes([DP[6].get_position().x1+0.01,DP[6].get_position().y0,
                 0.01,DP[6].get_position().height])                      
cax2=fig.add_axes([DP[7].get_position().x1+0.01,DP[7].get_position().y0, 
                 0.01,DP[7].get_position().height])

# Trazar 
fig.colorbar(o, cax=cax1,label="[mm/día]")
fig.colorbar(p, cax=cax2,label="[mm/día]")
plt.show

#*****************************************************************************************#
#                            DESCARGA DE DATOS - RAIN4PE                 
#*****************************************************************************************#
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import cartopy.feature as cfeature

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']},size=12)
rc('text', usetex=False)

# Data diaria
ds = xr.open_dataset("D:/R/RAIN4PE_daily_0.1d_1981_2015_v1.0.nc")
print(ds)
# Descarga para un punto
obs= ds.sel(Latitude=-12.958, Longitude= -74.063, method= 'nearest')
obs2 =obs.to_dataframe()
obs3= obs2['pcp']
obs3.plot()

# Descarga para varios puntos (Estaciones)

lista = [
    {'nombre': 'Est_V1', 'Longitude': -74.189, 'Latitude': -13.053},
    {'nombre': 'Est_V2', 'Longitude': -74.382, 'Latitude': -13.086},
    {'nombre': 'Est_V3', 'Longitude': -74.397, 'Latitude': -13.269},
    {'nombre': 'Est_V4', 'Longitude': -74.05, 'Latitude': -13.23},
    {'nombre': 'Est_V5', 'Longitude': -74.518, 'Latitude': -13.236},
    {'nombre': 'Est_V6', 'Longitude': -74.201, 'Latitude': -13.245},
    {'nombre': 'Est_V7', 'Longitude': -74.294, 'Latitude': -13.169},
    {'nombre': 'Est_V8', 'Longitude': -74.594, 'Latitude': -13.349},
    {'nombre': 'Est_V9', 'Longitude': -74.382, 'Latitude': -13.42},
    {'nombre': 'Est_V10', 'Longitude': -74.232, 'Latitude': -13.947},
]  

# Ubicamos la data
prec = xr.Dataset()
prec_ds = ds

'''Crear un conjunto de datos que contiene 
los valores de Precipitación para cada ubicación'''
for l in lista:
    nombre = l['nombre']
    Longitude = l['Longitude']
    Latitude = l['Latitude']
    var_name = nombre

    ds2 = prec_ds .sel(Longitude=Longitude, Latitude=Latitude, method='nearest')

    Longitude_attr = '%s_Longitude' % nombre
    Latitude_attr = '%s_Latitude' % nombre

    ds2.attrs[Longitude_attr] = ds2.Longitude.values.tolist()
    ds2.attrs[Latitude_attr] = ds2.Latitude.values.tolist()
    ds2 = ds2.rename({'pcp' : var_name}).drop(('Latitude', 'Longitude'))

    prec = xr.merge([prec, ds2], compat='override')

prec.data_vars

# Convertir a DataFrame
df_f = prec.to_dataframe() 
#df_f.describe()

# Plot 1
ax = df_f.plot(figsize=(20, 8), title="$Precipitación$", grid=2)
ax.set(xlabel='Date', ylabel='$mm\,dia^{-1}$')
#plt.axhline(y=20, c='gray', ls=':')
#plt.axvline(x=1995, c='gray', ls='--')
#plt.axvline(x=1989, c='gray', ls='--')
leg = plt.legend(facecolor='black', framealpha=0.8)
for text in leg.get_texts():
    plt.setp(text, color = 'w')
ax = plt.gca()
ax.set_facecolor('#e6fafa')
plt.text(0.01, 0.98 ,"Cuenca Cachi", fontsize=14, 
         transform=ax.transAxes, verticalalignment='top', color='blue')
plt.text(0.90, 0.98 ,"RAIN4PE", fontsize=14, 
         transform=ax.transAxes, verticalalignment='top', color='red')

# Plot 2
plt.rcParams["figure.figsize"] = (15,8) # Ancho*Altura
df_f.plot(subplots=True); plt.legend(loc="best")
plt.xticks(size="small",color="blue", rotation = 45)
plt.xlabel("Date")
#plt.text(100, 600, 'Antlantic Ocean', color = 'white', fontfamily = 'serif', fontsize = 'x-large')
plt.ylabel("")
#df_f.to_csv("D:/Qswat_cachi/Jonatan_tesis/Scenarios/Simulacion_Python/RAIN4PE/RAIN4PE_Vinchos.csv", index=True)

df_f.to_csv("D:/Paper_Climate/Paper/RAIN4PE/RAIN4PE_Cachi.csv", index=True)


#*****************************************************************************************#
#                            PLOT ANIMACIÓN - RAIN4PE                 
#*****************************************************************************************#
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

hp_shp = Reader("D:/T_JONA/TESIS_PISCO/DEPARTAMENTOS.shp")

prec = ds.pcp
fig = plt.figure()
(prec[0,:,:]).plot.contourf(levels=70)

fig = plt.figure(figsize=[8,8])

# Map con cartopy
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines(linewidths=0.5)

(prec[0,:,:]).plot.contourf(ax=ax,
                           transform=ccrs.PlateCarree(),
                           levels=70,
                           cmap="jet",
                           cbar_kwargs={
                               "orientation": "horizontal",
                               "label": "$pcp [mm/día]$",
                               "ticks": np.arange(0, 70, 30),
                               "shrink": 0.40}
                            )
# configurar ejes
ax.set_xlabel("")
ax.set_xticks(np.arange(85, 67, 10))

ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(ShapelyFeature(hp_shp.geometries(),ccrs.PlateCarree(), edgecolor='black'), facecolor='none')

ax.set_ylabel("")
ax.set_yticks(np.arange(1, -18, 0.5))

from matplotlib.animation import FuncAnimation

def update(i):
    prec[i, :, :].plot.contourf(ax=ax,
                       transform=ccrs.PlateCarree(),
                       levels=70,
                       cmap="jet",
                       add_colorbar=False
                        )

    ax.set_xlabel("")
    #ax.set_xticks(np.arange(-180, 181, 30))
    ax.set_ylabel("")
    #ax.set_yticks(np.arange(-90, 91, 30))
    # título de la figura
    ax.set_title(" RAIN4PE " + str(i+1));
    
# run !!!
anim = FuncAnimation(fig, update, frames=30)
#anim = animation.FuncAnimation(fig, update, frames=1140, interval=200)

# Guardar !!!
anim.save('D:/Qswat_cachi/Jonatan_tesis/Scenarios/Simulacion_Python/RAIN4PE_30.gif', writer='pillow', fps=5, dpi=300)