# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 17:51:13 2024

@author: ASUS
"""
import time
import xarray as xr
import numpy as np
from bias_correct import BiasCorrectDaily, convert_to_float32

# Rutas de archivos
f_observed = 'D:/Y/prism_example.nc'  # Ruta del archivo de datos observados
f_modeled = 'D:/Y/merra_example_mm_day.nc'   # Ruta del archivo de datos modelados
obs_var = 'ppt'                        # Nombre de la variable en los datos observados
modeled_var = 'PRECTOTLAND'            # Nombre de la variable en los datos modelados
output_file = 'D:/Y/output_corrected.nc'  # Archivo para guardar el resultado

# Cargar los datos observados
print("Loading observed data from:", f_observed)
obs_data = xr.open_dataset(f_observed)

print("Loading observations...")
obs_data.load()
obs_data = obs_data.dropna('time', how='all')  # Eliminar tiempos con todos los NaN

# Resample daily mean
obs_data = obs_data.resample(time='D').mean()  # Especificar 'time' como argumento

# Convertir a float32
obs_data = convert_to_float32(obs_data)

# Cargar los datos modelados
print("Loading modeled data from:", f_modeled)
modeled_data = xr.open_dataset(f_modeled)
if 'time_bnds' in modeled_data:
    del modeled_data['time_bnds']  # Eliminar 'time_bnds' si existe

modeled_data.load()
modeled_data = modeled_data.dropna('time', how='all')  # Eliminar tiempos con todos los NaN

# Resample daily mean
modeled_data = modeled_data.resample(time='D').mean()  # Especificar 'time' como argumento
modeled_data = convert_to_float32(modeled_data)

# Corrección de sesgo
print("Starting bias correction...")
t0 = time.time()
bc = BiasCorrectDaily(max_train_year=2001, pool=15)  # Puedes ajustar el número de trabajos
corrected = bc.bias_correction(obs_data, modeled_data, obs_var, modeled_var, njobs=1)
print("Running time:", (time.time() - t0))

# Guardar el dataset corregido
corrected.to_netcdf(output_file)
print("Bias corrected dataset saved to:", output_file)

##############   data scaling #######################

# Resample daily mean
obs_data = obs_data.resample(time='D').mean()  # Especificar 'time' como argumento
obs_data = convert_to_float32(obs_data)

# Resample daily mean
modeled_data = modeled_data.resample(time='D').mean()  # Especificar 'time' como argumento
modeled_data = convert_to_float32(modeled_data)

# Cálculo de factores de escalado
print("Calculating scaling factors...")
# Asegurarse de que los datos de observación y modelo están alineados por tiempo
obs_data = obs_data.sel(time=modeled_data['time'].values, method='nearest')

# Calcular factores de escalado
scale_factors = obs_data.groupby('time.dayofyear').mean(dim='time') / modeled_data.groupby('time.dayofyear').mean(dim='time')

# Guardar los factores de escalado en un archivo NetCDF
scale_factors.to_netcdf('D:/Y/archivo_de_escalado.nc')
print("Scaling factors saved to 'D:/Y/archivo_de_escalado.nc'")

# Ejemplo de uso de los factores de escalado
print("Applying scaling factors...")
scale_factors = xr.open_dataset('D:/Y/archivo_de_escalado.nc')

import xarray as xr
import argparse

# parse arguments 解析参数
# parse arguments 解析参数
parser = argparse.ArgumentParser()
parser.add_argument('bias_corrected', help="The bias corrected gcm or reanalysis file.")
parser.add_argument('scale_file', help="Netcdf file withe scaling factors.")
parser.add_argument('fout', help='BCSD output file')
args = parser.parse_args()
args = vars(args)

scale = xr.open_dataset(args['scale_file'])
bc = xr.open_dataset(args['bias_corrected'])


scaledayofyear = scale['time.dayofyear']

# align indices
print("Grouping")
scale = scale.groupby('time.dayofyear').mean('time')
scale['lat'] = bc.lat
scale['lon'] = bc.lon

daydata = []
for key, val in bc.groupby('time.dayofyear'):
    # print(key)
    # multiply interpolated by scaling factor
    if key == 366:
        key = 365
    daydata += [val.bias_corrected * scale.sel(dayofyear=key)]

# join all days
bcsd = xr.concat(daydata, 'time')
bcsd = bcsd.sortby('time')

bcsd.to_netcdf(args['fout'])

import pandas as pd

# Convertir a DataFrame y obtener descripción
obs_df = obs_data[obs_var].to_dataframe().reset_index()
modeled_df = modeled_data[modeled_var].to_dataframe().reset_index()

print("Descripción de los datos observados:")
print(obs_df.describe())

print("Descripción de los datos modelados:")
print(modeled_df.describe())

corrected.isel(time=0).to_netcdf("D:/Y/test_output.nc")


dset = xr.open_dataset("D:/Y/prism_example.nc")
dset1 = xr.open_dataset("D:/Y/merra_example_mm_day.nc")

print(dset1)
print(dset['ppt'])
print(dset["ppt"].attrs)

# Seleccionar la variable de precipitación
precip_var = "PRECTOTLAND"  # Cambia esto si tu variable de precipitación tiene otro nombre

# Realizar la conversión a mm/día
if precip_var in dset1:
    precip_data = dset1[precip_var] * 86400  # Multiplicar por 86400 para convertir a mm/día
    precip_data.attrs["units"] = "mm/day"  # Actualizar las unidades

    # Reemplazar la variable en el dataset con la convertida
    dset1[precip_var] = precip_data

    # Guardar el archivo actualizado (opcional)
    dset1.to_netcdf("D:/Y/merra_example_mm_day.nc")
    print("Conversión completada y guardada como 'merra_example_mm_day.nc'")
else:
    print(f"La variable {precip_var} no se encuentra en el dataset.")


print("Unidades después de la conversión:", dset1[precip_var].attrs["units"])










