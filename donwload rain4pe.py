# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 14:39:05 2024

@author: ASUS
"""
import xarray as xr
import matplotlib.pyplot as plt
# Data diaria
ds = xr.open_dataset("D:/R/RAIN4PE_daily_0.1d_1981_2015_v1.0.nc", engine="netcdf4")

# Data mensual
# ds = xr.open_dataset("D:/T_JONA/TESIS_PISCO/Entrada/Pisco_Pp/PISCOpm.nc", engine="netcdf4",decode_times=False, use_cftime=True)#Para data mensual

# Vemos que contiene la data
print(ds)
# Descarga de datos de la variable (precipitación) para la ubicación de las estaciones
# Crear un diccionario
locs = [
    {'nombre': 'Est_V1', 'Longitude': -74.57, 'Latitude': -13.42},
    {'nombre': 'Est_V2', 'Longitude': -74.52, 'Latitude': -13.42},
    {'nombre': 'Est_V3', 'Longitude': -74.57, 'Latitude': -13.37},
    {'nombre': 'Est_V4', 'Longitude': -74.52, 'Latitude': -13.37},
    {'nombre': 'Est_V5', 'Longitude': -74.47, 'Latitude': -13.37},

]
# Ubicamos la data
prec = xr.Dataset()
prec_ds = ds

# Crear un conjunto de datos que contiene los valores de Precipitación para cada ubicación
for l in locs:
    nombre = l['nombre']
    Longitude = l['Longitude']
    Latitude = l["Latitude"]
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
df_f = prec.to_dataframe() # Precipitación diaria
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
ax.set_facecolor('#faf5e6')
plt.text(0.01, 0.98 ,"Cuenca Cachi", fontsize=14, transform=ax.transAxes, verticalalignment='top', color='blue')

# Guardar datos!!!
df_f.to_csv("D:/L/LIMACO/piscopd_RANI4PE.csv", index=True)


