# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 12:36:47 2023

@author: ASUS
"""

import xarray as xr
import matplotlib.pyplot as plt


from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']},size=12)
rc('text', usetex=False)

# Data diaria
ds = xr.open_dataset("D:/R/RAIN4PE_daily_0.1d_1981_2015_v1.0.nc")

ds2012_mon = ds.groupby('time.month').sum()
ds2012_mon.pcp[0,:,:].plot(cmap='jet', vmax=300)

import calendar
landmask = ds.pcp.sum(dim='time')>0

fig = plt.figure(figsize=[12,8], facecolor='w')
plt.subplots_adjust(bottom=0.15, top=0.96, left=0.04, right=0.99, 
                    wspace=0.2, hspace=0.27) # wspace and hspace adjust the horizontal and vertical spaces, respectively.
nrows = 3
ncols = 4
for i in range(1, 13):
    plt.subplot(nrows, ncols, i)
    dataplot = ds2012_mon.pcp[i-1, :, :].where(landmask) # Remember that in Python, the data index starts at 0, but the subplot index start at 1.
    p = plt.pcolormesh(ds2012_mon.Longitude, ds2012_mon.Latitude, dataplot,
                   vmax = 250, vmin = 0, cmap = 'jet',
                   ) 
    plt.xlim([-83,-66])
    plt.ylim([-17,1])
    plt.title(calendar.month_name[dataplot.month.values], fontsize = 13, 
              fontweight = 'bold', color = 'b')
    plt.xticks(fontsize = 11)
    plt.yticks(fontsize = 11)
    if i % ncols == 1: # Add ylabel for the very left subplots
        plt.ylabel('Latitude', fontsize = 11, fontweight = 'bold')
    if i > ncols*(nrows-1): # Add xlabel for the bottom row subplots
        plt.xlabel('Longitude', fontsize = 11, fontweight = 'bold')

# Add a colorbar at the bottom:
cax = fig.add_axes([0.25, 0.06, 0.5, 0.018])
cb = plt.colorbar(cax=cax, orientation='horizontal', extend = 'max',)
cb.ax.tick_params(labelsize=11)
cb.set_label(label='Precipitation (mm)', color = 'k', size=14)


# Definir landmask
# Definir landmask
# Definir landmask
landmask = ds.pcp.sum(dim='time') > 0

# Configurar la figura
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=[12, 8], facecolor='w')
plt.subplots_adjust(bottom=0.15, top=0.96, left=0.04, right=0.99,
                    wspace=0.2, hspace=0.27)

# Iterar sobre los meses usando enumerate
for i, ax in enumerate(axes.flatten(), start=1):
    dataplot = ds2012_mon.pcp[i - 1, :, :].where(landmask)
    
    p = ax.pcolormesh(ds2012_mon.Longitude, ds2012_mon.Latitude, dataplot,
                      vmax=250, vmin=0, cmap='viridis')  # Usar 'viridis' colormap
                      
    ax.set_xlim([-83, -66])
    ax.set_ylim([-17, 1])
    ax.set_title(calendar.month_name[i], fontsize=13, fontweight='bold', color='b')
    ax.tick_params(axis='both', labelsize=11)  # Usar tick_params para configurar las etiquetas de los ejes
    
    if i % ncols == 1:
        ax.set_ylabel('Latitude', fontsize=11, fontweight='bold')
        
    if i > ncols * (nrows - 1):
        ax.set_xlabel('Longitude', fontsize=11, fontweight='bold')

# Agregar colorbar
cax = fig.add_axes([0.25, 0.06, 0.5, 0.018])
cb = plt.colorbar(p, cax=cax, orientation='horizontal', extend='max')
cb.ax.tick_params(labelsize=11)
cb.set_label(label='Precipitation (mm)', color='k', size=14)

# Agregar título principal
plt.suptitle('Monthly Precipitation Data for 2012', fontsize=16, fontweight='bold')

# Agregar barra de escala
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(axes[-1, -1])
cax_scale = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(p, cax=cax_scale, label='mm', extend='max')

plt.show()
#Asegúrate de definir adecuadamente las variables ds2012_mon y ds, ya que no las proporcionaste en el código original. También puedes personalizar más aspectos, como los límites de los ejes, los mapas de colores y las posiciones de las leyendas, según tus necesidades específicas.







