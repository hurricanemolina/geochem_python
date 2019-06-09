#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 16:21:13 2019

Maria Molina
"""

#########################################################################################
#########################################################################################
#########################################################################################


import xarray as xr
import matplotlib.pyplot as plt


#########################################################################################
#########################################################################################
#########################################################################################


#import your data set in netcdf format
data = xr.open_dataset('file1_new_stephan.nc', decode_cf=True)


#########################################################################################
#########################################################################################
#########################################################################################


#extract data variables and limit all the data to the first 7-time points.
time_values = data.time[:7].values

Ao_values = data.A_o[:7].values
A = data.A[:7].values
B = data.B[:7].values
C = data.C[:7].values
D = data.D[:7].values


#########################################################################################
#########################################################################################
#########################################################################################


#make figure layout
fig = plt.figure(figsize=(8.,4.))

#set axes for the plot within the figure
ax = fig.add_axes([0.0, 0., 1., 1.]) 

#the line plots, time versus data
ls1, = ax.plot(time_values, Ao_values, c='grey', zorder=1)
ls2, = ax.plot(time_values, A, c='goldenrod', zorder=2)
ls3, = ax.plot(time_values, B, c='chocolate', zorder=3)
ls4, = ax.plot(time_values, C, c='royalblue', zorder=4)
ls5, = ax.plot(time_values, D, c='dodgerblue', zorder=5)

#the scatter points of data observations
ax.scatter(time_values, Ao_values, c='grey', zorder=1)
ax.scatter(time_values, A, c='goldenrod', zorder=2)
ax.scatter(time_values, B, c='chocolate', zorder=3)
ax.scatter(time_values, C, c='royalblue', zorder=4)
ax.scatter(time_values, D, c='dodgerblue', zorder=5)

#force plot to not have any buffer space
plt.margins(x=0, y=0)

#the plot legend and other plot labels
ax.legend([ls1, ls2, ls3, ls4, ls5],
          ['MoO$_{4}$','MoO$_{3}$S','MoOS$_{3}$','MoO$_{2}$S$_{2}$','MoS$_{4}$'],
                              loc='upper right',shadow=True, 
          fancybox=True, ncol=2, fontsize=12, framealpha=1.)

ax.set_xlabel(r'Time (s; $ \times 10^{5}$)', fontsize=12, color='k')
ax.set_ylabel(r'[Mo] ($\mu$m; $ \times 10^{-5}$)', fontsize=12, color='k')

ax.set_title('Measured [Mo] over Time', fontsize=12, color='k')

ax.set_xticks([0.,100000,200000,300000,400000])
ax.set_xticklabels(['0','1','2','3','4'], fontsize=12)

ax.set_yticks([0.,0.00001,0.00002,0.00003,0.00004,0.00005])
ax.set_yticklabels(['0','1','2','3','4','5'], fontsize=12)

ax.grid(which='major', axis='both', linestyle='--', alpha=0.5)

#save your image
plt.savefig('image_steph3.png', bbox_inches='tight', pad_inches=0.075, dpi=200)
plt.close()



#########################################################################################
#########################################################################################
#########################################################################################

