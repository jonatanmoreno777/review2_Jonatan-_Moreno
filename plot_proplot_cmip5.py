# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 20:47:14 2021

@author: ASUS
"""

import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt
import numpy as np
import cmocean
import argparse
import seaborn as sns
#import cdutil
import iris
#from gamap_colormap import WhGrYlRd 
#accesscm2_pr_file = "/Users/Desktop/data-carpentry/data/pr_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_201001-201412.nc"

dset = xr.open_dataset("C:/Users/ASUS/Desktop/data-carpentry/.nc/pr_SAM-20_CCCma-CanESM2_rcp85_r1i1p1_INPE-Eta_v1_day_20610101-20651231.nc")
#dset = xr.open_dataset("D:/descarga _esgf/CIMP5_2006_2100/pr_Amon_CanESM2_rcp85_r1i1p1_200601-210012.nc")

print(dset)
dset.variables.keys()
clim = dset["pr"].mean("time" , keep_attrs=True)

#clim.data = clim.data * 86400

clim.data = clim.data* 86400
clim.attrs['units'] = '$mm\,dia^{-1}$'
#clim.attrs['units'] = '$kg\,m^{-2}\,s^{-1}$'
print((clim.data.max()),(clim.data.min()))#####baras


import proplot as plot

# (https://github.com/lukelbd/proplot/issues/79)
f, axs = plot.subplots(proj='robin', proj_kw={'lon_0':20}, width=8)#proj='cyl'

m = axs[0].contourf(clim, cmap='jet', levels=np.arange(0, 60.9, 5.5), extend='both')#cmap='ColdHot'

f.colorbar(m, label=clim.long_name + "["+clim.units+"]")

axs.format(
    geogridlinewidth=0.5, geogridcolor='gray8', geogridalpha=0.5, labels=True, 
    coast=True, ocean=True, oceancolor='gray3',
    suptitle="$ Modelo\ GCM, CMIP5 $"
)

####################3 temporadas###################################################


seas_CCCma = dset.groupby('time.season').mean()
seas_CCCma['pr']=seas_CCCma['pr']*86400
seas_CCCma.attrs['units'] = '$mm\,dia^{-1}$'


#seas_CCCma_xa = seas_CCCma.to_netcdf("D:/descarga _esgf/CIMP5_2006_2100/season_diff.nc")


season_clim_diff = xr.open_dataarray("D:/descarga _esgf/CIMP5_2006_2100/season_diff.nc")

f, axs = plot.subplots(
    proj='cyl', proj_kw={'lon_0':180}, ncols=2, nrows=2, axwidth=3, share=3
)

seasons = ['DJF', 'MAM', 'JJA', 'SON']

for i, ax in enumerate(axs):
    m = ax.contourf(
        season_clim_diff.sel(season=seasons[i]), cmap='ColdHot', norm='midpoint'
    )
    ax.format(title=season_clim_diff.sel(season=seasons[i]).season.values)
    
f.colorbar(m, label="Precipitación [mm/día]")
axs.format(
    geogridlinewidth=0.5, geogridcolor='gray8', geogridalpha=0.5, 
    labels=True, lonlines=60, latlines=30, 
    coast=True, ocean=True, oceancolor='gray4',
    suptitle="Temporada (seasons)", 
    abc=True, abcstyle='a.'
)

################################################################################

e1 = xr.open_dataarray("D:/descarga _esgf/CIMP5_2006_2100/season_diff.nc")
e2 = xr.open_dataarray("D:/descarga _esgf/CIMP5_2006_2100/season_diff.nc")
e3 = xr.open_dataarray("D:/descarga _esgf/CIMP5_2006_2100/season_diff.nc")
e4 = xr.open_dataarray("D:/descarga _esgf/CIMP5_2006_2100/season_diff.nc")


errs = [e1, e2, e3, e4]

titles = [
    'amip-bias-cor_vs_CNRM_198101-200012',
    'amip-bias-cor_vs_CNRM_208101-210012',
    'amip-free_vs_CNRM_198101-200012',
    'amip-free_vs_CNRM_198101-200012'
]

f, axs = plot.subplots(nrows=2, ncols=2, proj='robin', axwidth=4)

for i, ax in enumerate(axs):
    m = ax.contourf(
        errs[i][0], cmap='RdBu_r', globe=True,
#         colorbar='r', norm='midpoint', # pour une colorbar par plot
        levels=plot.arange(-100, 100, 10)        
    )
    ax.format(title=titles[i])
    
f.colorbar(m, label="blabla [%]")

axs.format(
    labels=True, # ne fonctionne pas encore avec Cartopy
    coast=True, geogridlinewidth=0.5, geogridcolor='gray8', geogridalpha=0.5,
    lonlines=60, latlines=30, 
    abc=True, abcstyle='a.',
    suptitle="zg500_CanESM2"
)



