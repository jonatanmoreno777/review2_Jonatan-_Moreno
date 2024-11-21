# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 11:25:18 2022

@author: ASUS
"""

import math
from math import tanh

def gr2m(precip, potential_evap, params, states = None, return_state = False):
    """
        Generated simulated streamflow for given rainfall and potential evaporation.
        :param precip: Catchment average rainfall.
        :type precip: array(float)
        :param potential_evap: Catchment average potential evapotranspiration.
        :type potential_evap: array(float)
        :param params: X parameters for the model.
        :type params: dictionary with keys X1, X5
        :param states: Optional initial state values.
        :type states: Dictionary with optional keys 'production_store', 'routing_store'.
        :param return_state: If true returns a dictionary containing 'production_store' and 'routing_store'. Default: False.
        :type return_state: boolean
        :return: Array of simulated streamflow.
    """

    X1 = params['X1']
    X2 = params['X2']

    if states is None:
        states = {}

    production_store = states.get('production_store', 0)
    routing_store = states.get('routing_store', 0)

    sims = []
    for P, PE in zip(precip, potential_evap):
        phi = tanh(P / X1)
        psi = tanh(PE / X1)

        S1 = (production_store + X1 * phi) / (1. + phi * (production_store / X1))

        P1 = P + production_store - S1

        S2 = S1 * (1 - psi) / (1 + psi * (1 - S1 / X1))

        production_store = S2 / pow(1. + pow(S2 / X1, 3), 1./3.)

        P2 = S2 - production_store

        P3 = P1 + P2

        R1 = routing_store + P3

        R2 = X2 * R1

        Q = pow(R2, 2) / (R2 + 60)
        sims.append(Q)

        routing_store = R2 - Q
    if return_state:
        return sims, {'production_store': production_store, 'routing_store': routing_store}
    else:
        return sims
    
import pandas as pd

# Leer el archivo CSV con Pandas
df = pd.read_csv('D:/T_JONA/TESIS_PISCO/Modelo GR2M/Data.csv')

# Obtener los datos de precipitación y evaporación potencial
precipitacion = df['P'].tolist()
evaporacion_potencial = df['E'].tolist()

# Definir los parámetros X1 y X2 (suponiendo que están definidos previamente)
params = {'X1': 650.7, 'X2': .8}

# Llamar a la función gr2m con los datos de precipitación, evaporación potencial y parámetros
flujo_simulado = gr2m(precipitacion, evaporacion_potencial, params)

# Imprimir el flujo simulado
print("Flujo simulado:", flujo_simulado)   


import matplotlib.pyplot as plt
# Crear una serie de fechas a partir del índice del DataFrame
# Agregar los datos simulados al DataFrame
df['Simulacion'] = flujo_simulado

# Crear el gráfico
ax = df.plot(figsize=(20, 8), title="Precipitación, Evaporación Potencial y Simulación de Flujo",
             grid=True, color=["green", "orange", "blue", "red"])

# Establecer etiquetas de ejes
ax.set(xlabel='Fecha', ylabel='Cantidad')

# Mostrar el gráfico
plt.show()



















    
    