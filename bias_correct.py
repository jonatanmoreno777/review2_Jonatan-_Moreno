# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:28:38 2024

@author: ASUS
"""
 
import pickle
import os, sys
import numpy as np
import xarray as xr
from joblib import Parallel, delayed

# Clase QMap para realizar el ajuste de cuantiles
class QMap():
    def __init__(self, step=0.01):
        self.step = step

    def fit(self, x, y, axis=None):
        if axis not in (None, 0):
            raise ValueError("Axis should be None or 0")
        self.axis = axis
        steps = np.arange(0, 100, self.step)
        self.x_map = np.percentile(x, steps, axis=axis)
        self.y_map = np.percentile(y, steps, axis=axis)
        return self

    def predict(self, y):
        idx = [np.abs(val - self.y_map).argmin(axis=self.axis) for val in y]
        if self.axis == 0:
            out = np.asarray([self.x_map[k, range(y.shape[1])] for k in idx])
        else:
            out = self.x_map[idx]
        return out

# Funciones auxiliares
np.seterr(invalid='ignore')

def mapper(x, y, train_num, step=0.01):
    qmap = QMap(step=step)
    qmap.fit(x[:train_num], y[:train_num], axis=0)
    return qmap.predict(y)

def nanarray(size):
    arr = np.empty(size)
    arr[:] = np.nan
    return arr

def convert_to_float32(ds):
    for var in ds.data_vars:
        if ds[var].dtype == 'float64':
            ds[var] = ds[var].astype('float32', copy=False)
    return ds

# Clase principal para corrección de sesgo
class BiasCorrectDaily():
    def __init__(self, pool=15, max_train_year=np.inf, step=0.1):
        self.pool = pool
        self.max_train_year = max_train_year
        self.step = step

    def bias_correction(self, obs, modeled, obs_var, modeled_var, njobs=1):
        d1 = obs.time.values
        d2 = modeled.time.values
        intersection = np.intersect1d(d1, d2)
        obs = obs.loc[dict(time=intersection)]
        modeled = modeled.loc[dict(time=intersection)]

        dayofyear = obs['time.dayofyear']
        lat_vals = modeled.lat.values
        lon_vals = modeled.lon.values

        mapped_data = np.zeros((intersection.shape[0], lat_vals.shape[0], lon_vals.shape[0]))

        for day in np.unique(dayofyear.values):
            print("Day = %i" % day)
            dayrange = (np.arange(day - self.pool, day + self.pool + 1) + 366) % 366 + 1
            days = np.in1d(dayofyear, dayrange)
            subobs = obs.loc[dict(time=days)]
            submodeled = modeled.loc[dict(time=days)]

            sub_curr_day_rows = np.where(day == subobs['time.dayofyear'].values)[0]
            curr_day_rows = np.where(day == obs['time.dayofyear'].values)[0]
            train_num = np.where(subobs['time.year'] <= self.max_train_year)[0][-1]
            mapped_times = subobs['time'].values[sub_curr_day_rows]
            
            jobs = []

            for i, lat in enumerate(lat_vals):
                X_lat = subobs.sel(lat=lat, lon=lon_vals, method='nearest')[obs_var].values
                Y_lat = submodeled.sel(lat=lat, lon=lon_vals)[modeled_var].values
                jobs.append(delayed(mapper)(X_lat, Y_lat, train_num, self.step))

            # print("Running jobs", len(jobs))
            # select only those days which correspond to the current day of the year
            day_mapped = np.asarray(Parallel(n_jobs=njobs)(jobs))[:, sub_curr_day_rows]
            day_mapped = np.swapaxes(day_mapped, 0, 1)
            mapped_data[curr_day_rows, :, :] = day_mapped

        dr = xr.DataArray(mapped_data, coords=[obs['time'].values, lat_vals, lon_vals],
                          dims=['time', 'lat', 'lon'])
        dr.attrs['gridtype'] = 'latlon'
        ds = xr.Dataset({'bias_corrected': dr})
        ds = ds.reindex_like(modeled)
        modeled = modeled.merge(ds)
        del modeled[modeled_var]
        return modeled

def test_qmap():
    np.random.seed(0)
    x = np.random.normal(10, size=(10, 20))
    y = np.random.normal(100, size=(10, 20))
    mapped = np.zeros(x.shape)
    for j in range(x.shape[1]):
        qmap = QMap()
        qmap.fit(x[:, j], y[:, j])
        mapped[:, j] = qmap.predict(y[:, j])
    print("Test QMap completado sin errores.")

# Ejecutar la prueba si el archivo se ejecuta como un script principal
if __name__ == "__main__":
    test_qmap()

    
    
