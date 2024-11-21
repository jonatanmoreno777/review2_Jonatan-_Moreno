# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 22:08:55 2021

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

# Enter file path for sim. and obs. flow files
simflowpath = 'C:/Users/ASUS/Desktop/data-carpentry/simu_flow.txt'
obsflowpath = 'C:/Users/ASUS/Desktop/data-carpentry/obs.txt'
# Enter file path for observed precipitation files
precipfilepath = 'C:/Users/ASUS/Desktop/data-carpentry/prec.txt'

# Read flow sim and obs dates &  data
simflowdata = np.genfromtxt(simflowpath, usecols=[1])
obsflowdata = np.genfromtxt(obsflowpath, usecols=[1])
tempsimflowdate = np.genfromtxt(simflowpath, dtype = str, usecols = 0)
tempobsflowdate = np.genfromtxt(obsflowpath, dtype = str, usecols = 0)
# Read obs. precipitation data
precipdata = np.genfromtxt(precipfilepath, dtype = float, usecols = [1])
##############################################
# Convert the string format dates into a datetime object
simflowdate = []
for i in range(len(tempsimflowdate)):
    simflowdate.append(datetime.datetime.strptime(tempsimflowdate[i],'%m/%d/%Y'))
# sort the dates in an acending order
simflowdate.sort()

# Convert the string format dates into a datetime object
obsflowdate = []
for i in range(len(tempobsflowdate)):
    obsflowdate.append(datetime.datetime.strptime(tempobsflowdate[i],'%m/%d/%Y'))
# sort the dates in an acending order
obsflowdate.sort()

# Extraiga las fechas coincidentes de ambas matrices y márquelas por máscara
# searchsorted (t1, t2) encuentra el índice en t1 para cada valor en t2:    
idx_flow = np.searchsorted(simflowdate, obsflowdate, side='right')-1
j2 = [i for i in idx_flow if i > 0]

# enmascarar la matriz como verdadera si 
mask_flow = idx_flow >=0

# compare the obs. and sim. time series
df = pd.DataFrame({"simflowdate":tempsimflowdate[idx_flow][mask_flow], "simflow":simflowdata[idx_flow][mask_flow], \
                   "obsflow":obsflowdata[mask_flow], "obsflowdate":tempobsflowdate[mask_flow]})
    
    
# average of sim
sim_mean= sum(simflowdata[idx_flow][mask_flow])/float(len(simflowdata[idx_flow][mask_flow]))
# average of obs
obs_mean= sum(obsflowdata[mask_flow])/float(len(obsflowdata[mask_flow]))
# total sum of squares 
ss_tot = sum((x-obs_mean)**2 for x in obsflowdata[mask_flow]) 
# sum of squares of residuals
ss_err = sum((y-x)**2 for y,x in zip(simflowdata[idx_flow][mask_flow],obsflowdata[mask_flow]))

nash_flow = 1 - (ss_err/ss_tot)
print ("nash_flow= ", nash_flow)

gradient, intercept, r_value, p_value, std_err = stats.linregress(simflowdata[idx_flow][mask_flow],obsflowdata[mask_flow])
print ("R-squared = ", r_value**2)
if gradient <= 1:
  adj_r2 = gradient * (r_value**2)
  print ("adj_r2 =", adj_r2)
if gradient > 1:
  adj_r2 = gradient**(-1) * (r_value**2)
  print ("adj_r2 =", adj_r2)
  
 ############################################################################## 
# reconstruct the date objects with matching dates only 
fdate=[]
for i in idx_flow:
  if i >= 0:
    fdate.append(datetime.datetime.strptime(tempsimflowdate[i],'%m/%d/%Y'))

ssim = pd.Series(simflowdata[idx_flow][mask_flow], index=fdate)
sobs = pd.Series(obsflowdata[mask_flow], index=fdate)

fig=plt.figure(figsize=(17,5),dpi=400)
ax = plt.subplot()
#fig.text(0.7, 0.8, r'$Nash=%.2f$' % nash_flow, fontsize=15)
plt.title('daily streamflow', fontsize=14, y=1.02, fontweight='bold')
plt.ylabel('flow. ($m^{3}\,s^{-1}$)', fontsize=12.)
plt.xlabel('year', fontsize=12.)

# s=square, -=dash, g^=triangle
#ssim['2009':'2016'].plot(label='Sim', color='red', linewidth=2.0)
#sobs['2009':'2016'].plot(label='Obs',color ='blue')
ssim['2009':'2016'].plot(label='Sim', color ='red', linewidth=2)
sobs['2009':'2016'].plot(label='Obs', color ='black', linewidth=1)#sobs['2009':'2016'].plot(label='Obs', color ='black', linewidth=1, style='-o', markersize=5)

ax.grid(True)
#tight_layout()
ax.legend(loc='best')


calstartdate = datetime.datetime(2009,1,1)
calenddate = datetime.datetime(2013,1,1)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
textstr1 = 'Calibration:\n$R^2 = 0.69$'
plt.text(0.6, 0.9, textstr1, fontsize=14, transform=ax.transAxes, verticalalignment='top', bbox=props)
textstr2 = 'Validation:\n$R^2 = 0.54$'
plt.text(0.4, 0.9, textstr2, fontsize=14, transform=ax.transAxes, verticalalignment='top', bbox=props)
threshold = 40
plt.ylim([0, 30])
ax.axvspan(calstartdate, calenddate, alpha=0.5, color='0.85')
#textstr3 = 'Validation:nash_flow=%.1f$^\circ$C\nR-squared=%.1f$^\circ$C\nnash_flow=%.2f   R-squared=%.2f' % (nash_flow[i], R-squared[i], nnash_flow[i], R-squared[i])
#ax.text(.02,.04, textstr3, fontsize=12, horizontalalignment='left', transform=ax.transAxes, bbox=props)

#########################################################################333
# contruct the time series
simdates = []
for i in range(len(tempsimflowdate)):
    simdates.append(datetime.datetime.strptime(tempsimflowdate[i],'%m/%d/%Y'))
ssim = pd.Series(simflowdata[:], index=simdates)

PCP = pd.Series(precipdata[:], index=simdates)

# Ingrese el rango de tiempo que desea trazar, el formato de fecha debe ser año, mes, dia 
startdate = datetime.datetime(2009,1,1)
enddate = datetime.datetime(2016,12,31)

# plot the primary sediment y-axis
fig = plt.figure(figsize=(12,4), dpi=400)
plt.title('3-hourly discharge', fontsize=14., y=1.02, fontweight='bold')
ax = fig.add_subplot(111)
pylab.xlim([startdate, enddate])
pylab.ylim([0,50])######### caudal max
#ax.set_xlabel("Year", fontsize=12.)
ax.set_ylabel(r"conc. ($m^{3}\,s^{-1}$)", fontsize=12.)
lns1=ax.plot(simdates, ssim, 'r', label='sim')
# plot the secondary ppt y-axis
ax2 = ax.twinx()
ax2= plt.gca()
ax2.set_ylabel(r"lluvia. ($mm\,dia^{-1}$)", fontsize=12.)
lns2=ax2.plot(simdates, PCP, 'g', label='precip')
pylab.xlim([startdate, enddate])
pylab.ylim([0,100])########pp max
ax2.invert_yaxis()
ax2.grid(True)
#add legend
lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc='best')
plt.show()


#https://github.com/jyearsley/PythonScripts/blob/1196248a6efb8cf87613fb70424cbf88586e19f1/StreamFlow.Daily.4Basins/STREAMFLOW_MERCER.ipynb

ssim = pd.Series(Fsimflow, index=Fsimflowdate)
sobs = pd.Series(Fobsflow, index=Fsimflowdate)
PCP = pd.Series(Fprecip, index=Fsimflowdate)

startdate = datetime.datetime(2009,1,1)
enddate = datetime.datetime(2016,12,31)

fig=plt.figure(figsize=(12,5),dpi=400)
ax = plt.subplot()
fig.text(0.7, 0.8, r'$Nash=%.2f$' % nash_flow, fontsize=15)
plt.title('daily streamflow @ 01144000', fontsize=14., y=1.02, fontweight='bold')
plt.ylabel('flow. ($m^{3}\,s^{-1}$)', fontsize=12.)
plt.xlabel('year', fontsize=12.)

#
lns1=ax.plot(Fsimflowdate, Fsimflow, 'r', label='sim')
lns2=ax.plot(Fobsflowdate, Fobsflow, 'b', label='obs')
plt.ylim([0,50])
# plot the secondary ppt y-axis
ax2 = ax.twinx()
ax2= plt.gca()
lns3 = ax2.bar(Fsimflowdate, PCP, 2)
ax2.grid(True)
ax2.set_ylabel(r"rain. ($m$)", fontsize=12.)
pylab.ylim([0,100])
ax2.invert_yaxis()

pylab.xlim([startdate, enddate])
ax.grid(True)
ax.legend((lns1[0],lns2[0],lns3[0]), ('Obs', 'sim', 'precip'),loc=(0.75, 0.7))
ax.legend(loc='best')

calstartdate = datetime.datetime(2009,1,1)
calenddate = datetime.datetime(2013,1,1)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
textstr1 = 'Calibration:\n$R^2 = 0.69$'
plt.text(0.6, 0.9, textstr1, fontsize=14, transform=ax.transAxes, verticalalignment='top', bbox=props)
textstr2 = 'Validation:\n$R^2 = 0.54$'

plt.text(0.4, 0.9, textstr2, fontsize=14, transform=ax.transAxes, verticalalignment='top', bbox=props)
threshold = 40
#plt.ylim([0, 30])
ax.axvspan(calstartdate, calenddate, alpha=0.5, color='0.85')

from scipy.stats import norm

ssim = pd.Series(Fsimflow, index=Fsimflowdate)
sobs = pd.Series(Fobsflow, index=Fobsflowdate)

fig=plt.figure(figsize=(7,5),dpi=400)
ax = plt.subplot()
n = len(ssim)

# log-normal probability plot
d = np.linspace(1, n, num=n)/(n+1)
y = norm.ppf(d, 0, 1)

# create the axis ticks
p  = [0.001, 0.01, 0.05, 0.2, 0.5, 0.75, 0.90, 0.98, 0.999];
# relate var "p" with "y"
tick  = norm.ppf(p,0,1);
label = ['0.001','0.01','0.05','0.2','0.5','0.75','0.90','0.98','0.999'];
# recast the p to (0,1)

print (len(sobs))
print (len(ssim))

# sort the data in an ascending order
Fsimflowdate.sort()
Fobsflowdate.sort()
ax.plot(y,ssim, 'r+', label='sim')
ax.plot(y,sobs,'b*', label='obs')

# use numpoints option so the markeres don't appear twice in one legend
ax.legend(loc=(0.05,0.8),numpoints = 1)
plt.ylabel('flow. ($m^{3}\,s^{-1}$)', fontsize=12.)
plt.xlabel('cumulative prob.', fontsize=12.)
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.FixedLocator(tick))
ax.xaxis.set_major_formatter(ticker.FixedFormatter(label))
ax.grid(True)
#https://github.com/jyearsley/PythonScripts/blob/1196248a6efb8cf87613fb70424cbf88586e19f1/DHSVM-WQ-PAPER-ANALYSES/Plot.ipynb


seguir
https://github.com/jyearsley/PythonScripts/blob/1196248a6efb8cf87613fb70424cbf88586e19f1/WQ.TimeStamped.4Basins/THORNTON.ipynb

https://github.com/jyearsley/PythonScripts/blob/master/.ipynb_checkpoints/Example_DHSVM_Analyses-checkpoint.ipynb



