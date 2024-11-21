# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 15:01:38 2021

@author: ASUS
"""

#Import des librairies requises
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors

#traitement du MNT
from pysheds.grid import Grid
#creation des shapefiles
from shapely import geometry, ops
import fiona

#Lecture du fichier .TIF
grid = Grid.from_raster('D:/T_JONA/TESIS_PISCO/srtm_90_cg/DEM_demo.tif', data_name='dem')#########contar con dem proximo a la cuenca

dir(grid)
##Supresión del formante del cuadro del MNT (sobre evitar los efectos de los cortes fronterizos sobre el MNT)
elevDem=grid.dem[:-1,:-1]

print('Système de coordonnées du MNT:', grid.crs.srs)
print('Zone emprise du MNT:', grid.bbox)
print('Données manquantes identifiées par:', grid.nodata)#Datos faltantes identificados por
print("Taille du MNT (n Lignes,m Colonnes)", grid.shape)#Tamaño del DEM (n filas, m columnas)
print("Altitude maximale",np.max(grid), "rencontrée aux indices (L,C) suivants", np.where(grid == np.max(grid)))
print("Altitude minimale",np.min(grid), "rencontrée aux indices (L,C) suivants", np.where(grid == np.min(grid)))

# ubicación de la salida por sus coordenadas
x_exu, y_exu = -107.91663,27.83479############-74.324, -12.901 cachi
#identificación de la celda incluida la salida en la cuadrícula
col_ex, row_ex = grid.nearest_cell(x_exu, y_exu)
print("Numéro de colonne incluant l'exutoire",col_ex)#Número de columna incluida la salida
print("Numéro de ligne incluant l'exutoire",row_ex)#Número de línea incluida la salida

## Representación 2D "imagen" en matriz de repetición (índice fila / índice columna)

plt.figure(figsize=(9,8))
plt.imshow(elevDem, cmap="terrain")
plt.title("Topographie DEM_demo")
plt.xlabel('Index Colonne')
plt.ylabel('Index Ligne')
plt.colorbar(label="Altitude (m)")
plt.grid()
plt.show()

## Representación 2D "imagen" en coordenadas WGS84 (coordenadas nativas del DEM)

plt.figure(figsize=(9,8))
plt.imshow(elevDem, extent=grid.extent,cmap="terrain")
plt.title("Topographie DEM_demo")
plt.xlabel('Longitude (Deg Méridien)')
plt.ylabel('Latitude (Nord)')
plt.colorbar(label="Altitude (m)")
plt.grid()

#si desea recuperar una imagen del DTM
#plt.savefig('Topo_DEM.png')


## Presentación alternativa: curvas de altitud
np.min(elevDem)##nivel minimo
np.max(elevDem)##nivel maximo
plt.figure(figsize=(9,8))
ax = plt.contour(elevDem,extent=grid.extent, colors = "black", 
            levels = list(range(0, 2800, 150)))#######nivel de la altitud
plt.title("Topographie courbes iso")
plt.xlabel('Longitude (Deg Méridien)')
plt.ylabel('Latitude (Nord)')
plt.gca().set_aspect('equal', adjustable='box')
#atención para ser homogéneo con las otras representaciones es necesario invertir el eje y
plt.gca().invert_yaxis()
plt.clabel(ax,colors = 'red', fmt= '%.0f', inline = True)
plt.show()


## Parcelas topográficas de sección transversal que atraviesan la altitud máxima encontrada en el DEM

ztop = (np.where(elevDem == np.max(elevDem)))[0]

crossEW=elevDem[ztop[0],:]
xcross = np.arange(elevDem.shape[1])
crossNS=elevDem[:,ztop[1]]
ycross = np.arange(elevDem.shape[0])

plt.figure(figsize = (10, 3))
plt.subplot(1, 2, 1)
plt.plot(crossEW)
plt.fill_between(xcross, crossEW, np.min(crossEW))
plt.title("Topographie W-E (Altitude Max)")
plt.xlabel('Index colonne')
plt.ylabel('Altitude (m)')
plt.subplot(1, 2, 2)
plt.plot(crossNS)
plt.fill_between(ycross, crossNS, np.min(crossNS))
plt.title("Topographie N-S (Altitude Max)")
plt.xlabel('Index ligne')
plt.ylabel('Altitude (m)')
plt.show()

# Identificación de depresiones

depressions = grid.detect_depressions('dem')

# Remplissage des depressions
grid.fill_depressions(data='dem', out_name='flooded_dem')

# Relleno de depresiones
plt.figure(figsize=(9,8))
plt.imshow(depressions, extent=grid.extent,cmap='Spectral')
plt.colorbar(label='index')
plt.grid()
plt.title("Identification des dépressions")
plt.show()

# Depresiones posteriores al llenado
plt.figure(figsize=(9,8))
plt.imshow(grid.detect_depressions('flooded_dem'), extent=grid.extent,cmap='Spectral')
plt.colorbar(label='index')
plt.grid()
plt.title("Identification des dépressions post-traitement")
plt.show()


# Detectar áreas planas ('planos')

flats = grid.detect_flats('flooded_dem')

# Plot flats
plt.figure(figsize=(9,8))
plt.imshow(flats, extent=grid.extent)
plt.colorbar(label='index')
plt.title("Identification des zones planes")
plt.grid()


#Reconstrucción del DEM poscondicionado
#Este paso permite eliminar las ambigüedades en las áreas planas que probablemente crearían una ruptura en la continuidad del flujo.
grid.resolve_flats(data='flooded_dem', out_name='inflated_dem')

plt.figure(figsize=(9,8))
plt.imshow(grid.inflated_dem[:-1,:-1], extent=grid.extent,cmap="terrain")
plt.colorbar(label='Altitude (m)')
plt.title("MNT post-conditionné")

plt.grid()


#Diferencia entre DTM(DEM) inicial y DTM(DEM) reacondicionado

ecart=grid.inflated_dem-grid.dem

plt.figure(figsize=(9,8))
plt.imshow(ecart, extent=grid.extent,cmap="tab10")
plt.colorbar(label='Différence entre MNT (m)')
plt.grid()


#Explotación temática en hidrología: identificar una cuenca asociada a una salida y su red de caudales

# Creación de la máscara de dirección D8(8 direcciones)
#N    NE    E    SE    S    SW    W    NW
dirmap = (64,  128,  1,   2,    4,   8,    16,  32)

# Cálculo de direcciones de flujo D8
# -------------------------------------
grid.flowdir(data='inflated_dem', out_name='dir', dirmap=dirmap)
grid_dir=grid.view('dir')

# Creación del mapa de flujo con una leyenda correspondiente a la dirección D8
cmap = mpl.colors.ListedColormap(['blue','black','red','yellow','tan','teal','blueviolet','aqua'])
bounds=[1,1.8,2.2,4.4,8.8,16.4,32.5,64.5,129]###  limites
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

plt.figure(figsize=(9,8))
plt.imshow(grid_dir, extent=grid.extent,cmap = cmap,norm=norm)
plt.grid()
plt.colorbar(boundaries=bounds,ticks=[1,2,4,8,16,32,64,128],label='Direction écoulement D8')
plt.show()

# Determinación de la cuenca asociada a la salida

# ubicación de la salida por sus coordenadas
x_exu, y_exu = -107.91663,27.83479

# determinación de la cuenca
grid.catchment(data='dir', x=x_exu, y=y_exu, dirmap=dirmap, out_name='catch',
               recursionlimit=15000, xytype='label', nodata_out=0)## limite de recursividad

# Recorte del área de visualización al BV limitado
grid.clip_to('catch')

# creación de una variable BV(limitado) para usar en vistas
BV = grid.view('dem', nodata=np.nan)

plt.figure(figsize=(9,8))
plt.imshow(BV, extent=grid.extent,cmap="terrain")
plt.colorbar(label='Altitude (m)')
plt.grid()

#si queremos exportar el resultado en un nuevo archivo DEM ráster
#grid.to_raster(BV, 'clippedElevations_WGS84.tif')

# Cálculo de índices de acumulación

grid.accumulation(data='catch', dirmap=dirmap, pad_inplace=False, out_name='acc')

accView = grid.view('acc', nodata=np.nan)
plt.figure(figsize=(9,8))
plt.imshow(accView, extent=grid.extent, cmap='Spectral')
plt.colorbar(label='Index accumulation')
plt.grid()

#Cálculo de la red hidrográfica en el umbral de drenaje considerado

#variable en función del índice - umbral para la división de la red hidroeléctrica
branche10 = grid.extract_river_network(fdir='catch', acc='acc',
                                      threshold=10, dirmap=dirmap)
branche50 = grid.extract_river_network(fdir='catch', acc='acc',
                                      threshold=50, dirmap=dirmap)
branche500 = grid.extract_river_network(fdir='catch', acc='acc',
                                      threshold=500, dirmap=dirmap)
branche5000 = grid.extract_river_network(fdir='catch', acc='acc',
                                      threshold=5000, dirmap=dirmap)

#representación gráfica de la red hidroeléctrica (variable / título a modificar según el umbral deseado)
plt.figure(figsize=(9,8))
plt.imshow(BV, extent=grid.extent,cmap="binary")
plt.title("Réseau hydrographique - Seuil = 500")
plt.grid()            
for branch in branche500['features']:
     line = np.asarray(branch['geometry']['coordinates'])
     plt.plot(line[:, 0], line[:, 1])

streams = grid.extract_river_network('catch', 'acc', threshold=10, dirmap=dirmap)

# Cálculo de la distancia de cada celda a la salida (unidad = celda o 30 m x 30 m)
# -------------------------------------------
grid.flow_distance(data='catch', x=x_exu, y=y_exu, dirmap=dirmap,
                   out_name='dist', xytype='label')
distance=grid.view('dist')

plt.figure(figsize=(9,8))
plt.imshow(distance, cmap='Spectral')
plt.colorbar(label="distance à l'exutoire (unite = cellule soit 30 m)")
plt.title("Réseau hydrographique - Seuil = 500")
plt.grid()

#Exportación de resultados en formato SIGV

# Ruta del BV para el outlet (esto ya se hizo, pero bueno ...)
grid.catchment(data='dir', x=x_exu, y=y_exu, dirmap=dirmap, out_name='catch',
               recursionlimit=15000, xytype='label', nodata_out=0)

# Ajuste de ventana a BV
grid.clip_to('catch')

# Creación del objeto vectorial que luego será exportado
shapes = grid.polygonize()

#para obtener información, podemos representar el objeto vectorial
#notamos que es necesario hacer un bucle en todas las coordenadas que componen el objeto vectorial
fig, ax = plt.subplots(figsize=(6.5, 6.5))

for shape in shapes:
    coords = np.asarray(shape[0]['coordinates'][0])
    ax.plot(coords[:,0], coords[:,1], color='cyan')
    
ax.set_xlim(grid.bbox[0], grid.bbox[2])
ax.set_ylim(grid.bbox[1], grid.bbox[3])
ax.set_title('Limites du BV (vecteur)')


#le fichier s'appellera BV.shp
schema = {
    'geometry': 'Polygon',
    'properties': {'LABEL': 'float:16'}
}

with fiona.open('BV.shp', 'w',
                driver='ESRI Shapefile',
                crs=grid.crs.srs,
                schema=schema) as c:
    i = 0
    for shape, value in shapes:
        rec = {}
        rec['geometry'] = shape
        rec['properties'] = {'LABEL' : str(value)}
        rec['id'] = str(i)
        c.write(rec)
        i += 1
        
# grille d'accumulation
grid.accumulation(data='catch', dirmap=dirmap, pad_inplace=False, out_name='acc')
#réseau hydrographique, au seuil choisi
branches = grid.extract_river_network('catch', 'acc', threshold=50, dirmap=dirmap)

schema = {
    'geometry': 'LineString',
    'properties': {}
}

with fiona.open('rivers.shp', 'w',
                driver='ESRI Shapefile',
                crs=grid.crs.srs,
                schema=schema) as c:
    i = 0
    for branch in branches['features']:
        rec = {}
        rec['geometry'] = branch['geometry']
        rec['properties'] = {}
        rec['id'] = str(i)
        c.write(rec)
        i += 1
        
#https://github.com/larroque852/ENS1CARTO/blob/f9fabeab44443a6509b0bddba89f84a034236bb3/ENS1_MNT_HYDRO.ipynb