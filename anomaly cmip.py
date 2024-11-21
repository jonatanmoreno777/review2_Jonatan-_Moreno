# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 13:33:07 2021

@author: ASUS
"""
#https://climdatanalyst.blogspot.com/2020/08/download-ERA4-CMIP-CDS.html
#import cdsapi
import xarray as xr
import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.pyplot as plt
import seaborn as sns
import tarfile

# import datasets
df_prcp= xr.open_dataset("D:/descarga _esgf/CIMP5_2006_2100/pr_Amon_CanESM2_rcp85_r1i1p1_200601-210012.nc")

df_temp= xr.open_dataset("D:/descarga _esgf/CIMP5_2006_2100/tasmax_Amon_CanESM2_rcp85_r1i1p1_200601-210012.nc")

df_temp.tasmax.attrs
df_prcp.pr

# Converts Temperature data into Degree celsius
# Converts Convect
df_temp_deg=df_temp.tasmax-273.15
df_temp_deg.attrs = df_temp.tasmax.attrs
df_temp_deg["units"] = "deg C"
df_temp_deg

# Converts Precipitation flux data into mm/month
df_pre=df_prcp.pr*86400*30

#Monthly mean
monmean_temp=df_temp_deg.groupby('time.month').mean()#monmean_temp=df_temp_deg.groupby('time.month').mean().mean('expver')##cmip5
monmean_prcp=df_pre.groupby('time.month').mean()

monmean_temp
monmean_prcp

projection = ccrs.PlateCarree()
axes_class = (GeoAxes,dict(map_projection=projection))

longitude=monmean_temp.lon.values #longitude
latitude=monmean_temp.lat.values #latitude
month=monmean_temp.month.values
months=['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'] #

fig = plt.figure()
fig.set_size_inches(18.5, 10.5)
axgrid = AxesGrid(fig, 111, axes_class=axes_class,
                  nrows_ncols=(3, 4), 
                  axes_pad=0.3,
                  cbar_location='right',
                  cbar_mode='single',
                  cbar_pad=0.2,
                  cbar_size='3%',
                  label_mode='')

for n, ax in enumerate(axgrid):
    ax.set_xlim(-20,25)
    ax.set_ylim(0,25)
    ax.set_xticks(np.linspace(-20,25,4), crs=projection)
    ax.set_yticks(np.linspace(0,25, 4), crs=projection)
    ax.coastlines()
    p = ax.contourf(longitude, latitude, monmean_temp[n],
                        transform=projection,
                        cmap='RdBu_r')
    ax.text(20,20, str(months[n]),fontsize=18, ha='center')

d=axgrid.cbar_axes[0].colorbar(p)
d.set_label_text('Deg C')


# Make ticklabels on inner axes invisible
axes = np.reshape(axgrid, axgrid.get_geometry())
for ax in axes[:-1, :].flatten():
    ax.xaxis.set_tick_params(which='both', 
                             labelbottom=False, labeltop=False)
    
for ax in axes[:, 1:].flatten():
    ax.yaxis.set_tick_params(which='both', 
                             labelbottom=False, labeltop=False)
#plt.suptitle('Monthly Temperatuve over West-Africa',fontsize=20,y=1.04)
plt.show()



###############
interan_temp=df_temp.groupby('time.year').mean() #Annual mean
df = interan_temp.tasmax.to_dataframe()
dd=df.tasmax.groupby('year').mean() # Yearly mean
clim_temp=df.loc[2006:2020,].tasmax.groupby('year').mean().mean() # climatology mean
clim_sd=df.loc[2006:2020,].tasmax.groupby('year').mean().std()# climatology standard deviation

temp_anom=(dd-clim_temp)/clim_sd # anomaly compuatation



fig = plt.figure(figsize=(10,5))
cols=['red' if x <0 else 'blue' for x in temp_anom]
plt.bar(temp_anom.index[1:94],temp_anom[1:94],color=cols)##35 años de lluvia
plt.xlabel('Years'),plt.ylabel('Anomaly')
plt.title('Temperature standardised anomaly', loc='left', fontsize=15)
plt.title('2006-2100',loc='right', fontsize=15)
plt.show()