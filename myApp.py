# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 22:54:43 2024

@author: ASUS
"""

import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.figure_factory as ff
import plotly.express as px
import numpy as np


st.write(
    """
        # Ontario Regional Streamflow Model
        ### A Deep Learning Model for Flow Forecasting in Ontario's Catchments 
        #### Application developed by Taimoor Akhtar
        #### 

        This application uses interactive visualizations to illustrate the forecast accuracy of the Regional Streamflow Model 
        created for 300+ Ontario catchments. The Regional model is developed using a Recurrent Neural Network architecture (LSTM). 
        The model is developed for Greenland Consulting Services' THREATS platform for forecasting future flows in both gaged and 
        ungaged catchments. The model is also applicable for generating flow estimates at gaged locations for time-series imputation 
        and data gap-filling.

        ### Distribution Plot - Overall Model Accuracy
        The model is tested using a 16-fold cross-validation strategy, and model accuracy for all 300+ monitoring locations is recorded 
        using three accuracy / performance measures, i.e., NSE, R Squared and Bias. The overall model accuracy is summarized via the 
        distribution plot of NSE values across all 300 locations / catchments (for a 15 year validation period). provided below (please note
         that NSE values higher and close to 1 are desirable):   
    """
)

## Get data and plot first figure
@st.cache_data
def get_data():
    return gpd.read_file('D:/Paper_Climate/Data/siguiente paper/App/App_SWAT/LSTM/ontario_catchment_results_anusplin.shp')

gdf = get_data()

df = gdf[['catch_id', 'nse_v1', 'rf_foldid']]
df = df[df['nse_v1'] > -1]
group_labels = ['NSE']
fig = ff.create_distplot([df['nse_v1']], group_labels, bin_size=0.05)
fig.update_layout(title='NSE distribution across 300+ Catchemnts for the Ontario Streamflow Model', 
                  xaxis_title='NSE') 
st.plotly_chart(fig)

## Step 2 --- Bubble plot for understanding error sources
st.write(
    """
        ### Bubble Plot - Attributes of Failing Catchments
        The plot below attempts to understands model accuracy trends as functions of some salient catchment 
        features. The key catchment features being analyzed are catchment size (xaxis) and catchment road 
        density ration (size of bubbles). The plot below analyzes change in R squared and model bias with 
        changes in catchment size and road density. While trends are not distinct, model bias is high for 
        very small catchments and higher road densities adversely affect model accuracy both in terms of 
        R squared and bias (color scale).   
    """
)

df = gdf[['catch_id', 'rsq_v1', 'pbias_v1', 'fn_catch_a', 'road_densi']]
df['pbias_v1'] = np.where(df['pbias_v1'] > 50, 50, df['pbias_v1'])
df['pbias_v1'] = np.where(df['pbias_v1'] < -50, -50, df['pbias_v1'])
df.columns = ['Catchment ID', 'R Squared', 'Bias (%)', 'Catchment Area (sq_km)', 'Road Density']
df = df.round({
    'R Squared': 2,
    'Bias (%)': 1,
    'Catchment Area (sq_km)': 1,
    'Road Density': 1
})
fig = px.scatter(df, x="Catchment Area (sq_km)", y="R Squared",
                 size="Road Density", color="Bias (%)", color_continuous_scale=px.colors.diverging.Spectral_r,
                 hover_name="Catchment ID", log_x=True)
st.plotly_chart(fig)

## Step 2 --- Bubble plot for understanding error sources
st.write(
    """
        ### Map-based Accuracy Summary - R Squared
        While the bubble plot provides some valuable insights into the type of catchments for which 
        our regoinal deep learning model performs well, a further map-based analysis of catchments 
        can provide further insights into catchment areas and river basins where the model performs well 
        and not well. The following interactive map reports the R squared of different catchments. Since higher 
        R squared values are desired, it is observed here that catchments that have dams within (place tooltip 
        over a catchment to see additional catchment features including dam influenced area), generally have 
        poor performance. Thus it may be concluded that our regional model performs reasonably well for catchments 
        without dams and will less road densities (i.e., less human interventions) (NOTE: The map may take 30 seconds to 60 seconds when loading the first time).
    """
)

@st.cache_data
def get_map(_gdf):
    _gdf = _gdf.sort_values(by=['fn_catch_a'])
    _gdf = _gdf.round({
        'rsq_v1': 2,
        'pbias_v1': 1,
        'fn_catch_a': 1,
        'road_densi': 1,
        'dam_area_r': 1
    })

    fig = px.choropleth_mapbox(_gdf,
                               geojson=_gdf.geometry,
                               locations=_gdf.index,
                               color='rsq_v1',
                               range_color=(0, 1),
                               hover_name='catch_id',
                               hover_data=['fn_catch_a', 'pbias_v1', 'dam_area_r', 'road_densi'],
                               center={"lat": 43.5, "lon": -80.7073},
                               mapbox_style="open-street-map",
                               opacity=0.5,
                               zoom=6,
                               labels={
                                   'rsq_v1': 'R Squared',
                                   'fn_catch_a': 'Catchment Area (sq_km)',
                                   'pbias_v1': 'Bias (%)',
                                   'dam_area_r': 'Dam Area Ratio',
                                   'road_densi': 'Road Density'
                               })
    return fig

fig = get_map(gdf)

st.plotly_chart(fig)


import os

# Comando para ejecutar tu aplicación de Streamlit
os.system('streamlit run myApp.py')
