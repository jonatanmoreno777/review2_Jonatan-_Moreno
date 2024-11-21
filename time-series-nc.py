# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 08:26:14 2022

@author: ASUS
"""

# import python packages that allow for data reading, analysis, and plotting
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfe
import pandas as pd
https://github.com/cturnbull23/wxdatasci/blob/master/coding-challenges/weathermap/WeatherMap_Demo.ipynb
https://github.com/cturnbull23/Cryosphere-Climate-Systems
# Set desired file names and variable names into two seperate lists
fileList = ['D:/T_JONA/TESIS_PISCO/Entrada/Pisco_Pp/Precday.nc',
            'D:/T_JONA/TESIS_PISCO/Entrada/Pisco_Temp_day/tmax.nc',
            'D:/T_JONA/TESIS_PISCO/Entrada/Pisco_Temp_day/tmin.nc',
]

varList = ['variable', 'tmax', 'tmin']

# Read in the variables
for fileName, varName in zip(fileList, varList):
    readCmnd = varName + " = xr.open_dataset('"  + fileName + "', decode_times=False)"
    print(readCmnd)
   
variable = xr.open_dataset('D:/T_JONA/TESIS_PISCO/Entrada/Pisco_Pp/Precday.nc')
RAIN4PE = xr.open_dataset('D:/R/RAIN4PE_daily_0.1d_1981_2015_v1.0.nc', decode_times=False)

# Define new time dimension
variable['time'] = pd.date_range(start='1981-01-01', periods=35, freq='YS')
RAIN4PE['time'] = pd.date_range(start='1981-01-01', periods=35, freq='YS')

# Calculate annual SMB for full GrIS
smb = variable['variable'].sum(dim=('latitude','longitude')) # sums across x and y dimensions, units 

# Set up figure
fig, ax = plt.subplots(figsize=(15, 7.5))
year = np.arange(1981, 2016 + 1)

# Plot data
smbGT_plot = ax.plot(year, smb, '-', linewidth=2, label='SMB')

# Configure parameters
ax.set(xlim=[1981, 2016])
ax.grid(True)
legend = ax.legend(fontsize=14)
ax.set_ylabel('Mass flux (Gt yr$^{-1}$)', fontsize=16)
ax.tick_params(labelsize=16)
plt.title("Greenland Ice Sheet Annual Timeseries of SMB Variables - RACMO2.3p2", fontsize=16)

# Save figure
plt.savefig('./Figures/GrIS-SMB.png', bbox_inches='tight', facecolor='white', dpi=300)

