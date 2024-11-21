# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 14:43:12 2021

@author: ASUS
"""

import sys, os, datetime, time
import numpy as np
import pandas as pd
import numpy.ma as ma
import math
import matplotlib.dates as dates
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm
from scipy import stats
import matplotlib.pyplot as pylab
from matplotlib.dates import date2num, num2date 
from matplotlib.pyplot import figure, show
import collections
 #https://github.com/jyearsley/PythonScripts/blob/master/4-Urban-Basins-AGU-Figure/StreamTemperature.ipynb ### temperatura
#https://github.com/jyearsley/PythonScripts/blob/master/4-Urban-Basins-AGU-Figure/Streamflow.ipynb   ####caudal
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2
font = {'family' : 'sans-serif',
        'sans-serif' : 'Verdana',
        'weight' : 'medium',
        'size'   : '12'}
params1 = {
          'axes.labelsize': 12,
          'text.fontsize': 12,
          'xtick.labelsize': 12,
          'xtick.direction': 'out',
          'ytick.labelsize': 12,
          'legend.pad': 0.01,     # empty space around the legend box
          'legend.fontsize': 12,
          'legend.labelspacing':0.25,
          'font.size': 12,
          'font.style': 'normal',
          'axes.style': 'normal',
          'xtick.labelstyle': 'normal',
          }
mpl.RcParams.update(params1)
mpl.rc('font', **font)
plt.rc("xtick", direction="out")
plt.rc("ytick", direction="out")
plt.rc('legend',**{'fontsize':12})

#################
basin_name = ['Choccoro', 'Chicllarazo', 'Apacheta']
mainpath = {}
simtemperaturepath = {}
obstemperaturepath = {}

for i, n in enumerate(basin_name):
    mainpath[n] = 'C:\\Users\\ASUS\\Desktop\\data-carpentry\\'+str(basin_name[i])
    simtemperaturepath[n] = str(mainpath[n])+'\\simu_flow.txt'
    obstemperaturepath[n] = str(mainpath[n])+'\\obs.txt'
    

# Read flow sim and obs dates & data
    
obsflowdata = {}
simflowdata = {}
tempobsflowdate = collections.defaultdict(list)
tempsimflowdate = collections.defaultdict(list)

# Read flow sim and obs dates &  data
for i, n in enumerate(basin_name):
    simflowdata[n] = np.genfromtxt(simtemperaturepath[n], usecols=[1])
    obsflowdata[n] = np.genfromtxt(obstemperaturepath[n], usecols=[1])
    tempobsflowdate[n] = np.genfromtxt(simtemperaturepath[n], dtype=str, usecols=0)
    tempsimflowdate[n] = np.genfromtxt(obstemperaturepath[n], dtype=str, usecols =0)
    
    
simflowdate = collections.defaultdict(list)
obsflowdate = collections.defaultdict(list)
DailyPrecip = {}
DailySimFlow = {}
DailyObsFlow = {}
MonthlySimFlow = {}
MonthlyObsFlow = {}

for i, n in enumerate(basin_name):
    for j in range(len(tempobsflowdate[n])):
        obsflowdate[n].append(datetime.datetime.strptime(tempobsflowdate[n][j],'%m/%d/%Y'))
    for j in range(len(tempsimflowdate[n])):
        simflowdate[n].append(datetime.datetime.strptime(tempsimflowdate[n][j],'%m/%d/%Y'))

# daily mean flow (unit: m^3/s)   
for i,n in enumerate(basin_name):
    DailySimFlow[n] = pd.Series(simflowdata[n], index=simflowdate[n])
    DailyObsFlow[n] = pd.Series(obsflowdata[n], index=obsflowdate[n])
    
    
basin_name = ['Choccoro', 'Chicllarazo', 'Apacheta']
startdate = [datetime.datetime(2009,1,1), datetime.datetime(2012,1,1), datetime.datetime(2013,1,1)] 
enddate = [datetime.datetime(2016,12,31), datetime.datetime(2016,12,31), datetime.datetime(2016,12,31)]

# Truncate the data into desired time frame
for i, n in enumerate(basin_name):
        DailySimFlow[n] = DailySimFlow[n].truncate(before=startdate[i], after=enddate[i])
        DailyObsFlow[n] = DailyObsFlow[n].truncate(before=startdate[i], after=enddate[i])
        
        
# monthly mean flow for each year
for i,n in enumerate(basin_name):
    MonthlySimFlow[n] = DailySimFlow[n].resample(rule="M").mean()#MonthlySimFlow[n] = DailySimFlow[n].resample('M', how='mean')
    MonthlyObsFlow[n] = DailyObsFlow[n].resample(rule="M").mean()
    
# average monthly streamflow over the period
for i,n in enumerate(basin_name):
    MonthlySimFlow[n] = MonthlySimFlow[n].groupby(lambda x: x.month).mean()
    MonthlyObsFlow[n] = MonthlyObsFlow[n].groupby(lambda x: x.month).mean()
    
nash_flow = {}
for i, n in enumerate(basin_name):
    sim_mean= sum(DailySimFlow[n])/float(len(DailySimFlow[n]))
    print ('sim_mean: ',sim_mean)
    obs_mean= sum(DailyObsFlow[n])/float(len(DailyObsFlow[n]))
    print ('obs_mean ',obs_mean)
    ss_tot = sum((x-obs_mean)**2 for x in DailyObsFlow[n]) 
    ss_err = sum((y-x)**2 for y,x in zip(DailySimFlow[n], DailyObsFlow[n]))
    nash_flow[n] = 1 - (ss_err/ss_tot)
    print ("gage[%s] daily flow nash = %.2f" % (str(basin_name[i]), nash_flow[n]))
    
 
    
from scipy.stats import norm 
    
fig=plt.figure(figsize=(8,12),dpi=400)
jj = 1
for i, n in enumerate(basin_name):
    ax = plt.subplot2grid((len(basin_name), 3), (0+i, 0), colspan=2)
    fig.subplots_adjust(hspace = 0.3, wspace = 0.4)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr1 = '%s\nNash = %.2f' % (str(n), nash_flow[n])
    ax.text(.02,.78, textstr1, fontsize=12, horizontalalignment='left', transform=ax.transAxes, bbox=props)
    
    DailyObsFlow[n].plot(label='Obs', color='grey', linewidth=2, style='-', alpha=0.8)
    DailySimFlow[n].plot(label='Sim', color='blue', linewidth=2, style='-', alpha=0.8)
    ax.grid(True)
    if (i == 0):
        leg=ax.legend(loc=1, numpoints = 1)
        leg.get_frame().set_alpha(0.5)
    plt.ylabel('flow ($m^{3}/s$)', fontsize=12., labelpad=5)
    
    #################### Cumulative Distribution Function ####################
    jj = jj + 1
    ax = plt.subplot2grid((len(basin_name), 3), (0+i, 2), colspan=1)
    jj = jj + 1
    
    nn = len(tempsimflowdate[n])
    # log-normal probability plot
    d = np.linspace(1, nn, num=nn)/(nn+1)
    y = norm.ppf(d, 0, 1)
        
    # create the axis ticks
    p  = [0.01, 0.2, 0.5, 0.90, 0.999];
    
    # relate var "p" with "y"
    tick  = norm.ppf(p,0,1);
    label = ['0.01','0.2','0.5','0.90','0.999'];
    
    # sort the data in an ascending order
    tempsimflowdate[n].sort()
    tempobsflowdate[n] .sort()
    ax.plot(y, tempobsflowdate[n],'*', color = '0.9', alpha = 0.8, label='obs')
    ax.plot(y, tempsimflowdate[n], 'b+', label='sim')
    
    # use numpoints option so the markeres don't appear twice in one legend
    ax.xaxis.set_major_locator(ticker.FixedLocator(tick))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(label))
    ax.grid(True)
    
    # Changing the label's font-size
    ax.tick_params(axis='x', labelsize=11)
    
    if (i == 0):
        leg=ax.legend(loc='best', numpoints = 1)
        leg.get_frame().set_alpha(0.5)
    plt.ylabel('flow ($m^{3}/s$)', fontsize=12., labelpad=5)
    if (i == 1):
        plt.xlabel('probability', fontsize=12., labelpad=5)   
        
        
# Reorganize the data set to conform to the wateryear (ie. Oct-Sep)
for i, n in enumerate(basin_name):
    MonthlyObsFlow[n] = MonthlyObsFlow[n][9:].append(MonthlyObsFlow[n][:9])
    MonthlySimFlow[n] = MonthlySimFlow[n][9:].append(MonthlySimFlow[n][:9])
    

nash_flow = {}
for i, n in enumerate(basin_name):
    sim_mean= sum(MonthlySimFlow[n])/float(len(MonthlySimFlow[n]))
    obs_mean= sum(MonthlyObsFlow[n])/float(len(MonthlyObsFlow[n]))
    ss_tot = sum((x-obs_mean)**2 for x in MonthlyObsFlow[n]) 
    ss_err = sum((y-x)**2 for y,x in zip(MonthlySimFlow[n], MonthlyObsFlow[n]))
    nash_flow[n] = 1 - (ss_err/ss_tot)
    print ("gage[%s] monthly flow nash = %.2f" % (str(basin_name[i]), nash_flow[n]))
    
    
fig=plt.figure(figsize=(12,12),dpi=400)
for i, n in enumerate(basin_name):
    ax = plt.subplot(4, 2, i+1)
    fig.subplots_adjust(hspace = 0.3, wspace = 0.3)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    textstr1 = '%s\nNash = %.2f' % (str(n), nash_flow[n])
    
    ax.text(.02,.78, textstr1, fontsize=12, horizontalalignment='left', transform=ax.transAxes, bbox=props)
    plt.plot(range(1, 13), MonthlyObsFlow[n].values, color='grey', label='obs', linewidth=2.0, linestyle='-')
    plt.plot(range(1, 13), MonthlySimFlow[n].values, color='blue', label='sim', linewidth=2.0, linestyle='-')

    plt.ylabel('flow ($m^{3}/s$)', fontsize=12.)
    if (i == 2):
        plt.xlabel('month', fontsize=12.)
    plt.xlim([1,12])
    ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(1,13)))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(['Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep']))
    ax.grid(False)
    if (i == 0):
        ax.legend(loc='best', bbox_to_anchor=(1, 1))

#outfig = str(mainpath) +'MonthlyFlow.png'    
#plt.savefig(str(outfig),dpi=300, format = 'png',transparent='True')
    