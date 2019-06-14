#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 01:29:09 2018

Authors:

Stephan Hlohowskyj
and
Dr. Maria J. Molina

"""

###############################################################################
###############################################################################
###############################################################################  


#this import needed for me because I am still using python version 2.7...
from __future__ import division

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


###############################################################################
###############################################################################
###############################################################################  


number_of_samples = 1000


###############################################################################
###############################################################################
###############################################################################  


#k value ranges

#MARIA'S QUESTION: do we no longer need kiiii? noticed we don't have this variable in equations.

ki_low = 0.00005
kii_low = 0.000009 
kiii_low = 0.0000005 
#kiiii_low = 0.00000001 

ki_hi = 0.0005
kii_hi = 0.00005 
kiii_hi = 0.000009
#kiiii_hi = 0.0000005

ki_range = np.linspace(ki_low,ki_hi,10)
kii_range = np.linspace(kii_low,kii_hi,10)
kiii_range = np.linspace(kiii_low,kiii_hi,10)
#kiiii_range = np.linspace(kiiii_low,kiiii_hi,1000)

#note: no kv for E needed
#note: ki>kii>kiii>kiiii


###############################################################################
###############################################################################
############################################################################### 


#import your data set in netcdf format
data = xr.open_dataset('file1_new_stephan.nc', decode_cf=True)

#extract data variables and limit all the data to the first 7-time points.
time_values = data.time[:7].values
Ao_realvalues = data.A_o[:7].values


###############################################################################
###############################################################################
############################################################################### 


#extracting randomized k-values for 10,000 member randomization

#we are setting a seed for randomization for testing and ensuring we get same random results every time.
#we are bootstrapping, allowing for values to get called more than once with numpy's random choice

np.random.seed(0)
rand_ki = np.array([np.random.choice(10) for i in xrange(number_of_samples)])

np.random.seed(1)
rand_kii = np.array([np.random.choice(10) for i in xrange(number_of_samples)])

np.random.seed(2)
rand_kiii = np.array([np.random.choice(10) for i in xrange(number_of_samples)])

#np.random.seed(3)
#rand_kiiii = np.array([np.random.choice(1000) for i in xrange(number_of_samples)])
    

ki_randomized = ki_range[rand_ki]
kii_randomized = kii_range[rand_kii]
kiii_randomized = kiii_range[rand_kiii]
#kiiii_randomized = kiiii_range[rand_kiiii]


###############################################################################
###############################################################################
############################################################################### 


#creating blank arrays to fill. 
A_solutions = np.zeros((number_of_samples,7))
B_solutions = np.zeros((number_of_samples,7))
C_solutions = np.zeros((number_of_samples,7))
D_solutions = np.zeros((number_of_samples,7))

for indexer, (ki, kii, kiii) in enumerate(zip(ki_randomized, kii_randomized, kiii_randomized)):

    for sec_indexer, (t, A_o) in enumerate(zip(time_values, Ao_realvalues)):
    
        # we will only have 4 equations since we are ignoring back reations, so the reactions proceed from A->D
    
        A = A_o * np.exp(-ki*t)
        
        
        B = np.divide((ki*A_o),(kii-ki)) * (np.exp(-ki*t)-np.exp(-kii*t))
    
        #Subject to change, Natalia and Dimtry will check my math for these equations
    
        C = np.divide((ki*A_o),(kii-ki)) * (((kiii-ki) * np.exp(-ki*t)) - ((kiii-kii) * np.exp(-kii*t)) + ((ki-kii) * np.exp(-kiii*t)))
        
        #MARIA'S COMMENT: Please double check bracket placements in equation for D!
        
        D = np.divide((ki*A_o*kiii), (kii-ki)) * \
                      ((np.divide((kiii-ki),(-ki)) * np.exp(-ki*t)) - \
                       (np.divide((kiii-kii),(-kii)) * np.exp(-kii*t)) + \
                       (np.divide((ki-kii),(-kiii)) * np.exp(-kiii*t))) - \
                       (np.divide((ki*A_o*kiii), (kii-ki)) * \
                        (np.divide((kiii-ki),(-ki)) - \
                         np.divide((kiii-kii),(-kii)) + \
                         np.divide((ki-kii),(-kiii))))
                    
        A_solutions[indexer, sec_indexer] = A
        B_solutions[indexer, sec_indexer] = B
        C_solutions[indexer, sec_indexer] = C
        D_solutions[indexer, sec_indexer] = D
        

    print indexer


#A_o = Tc = A + B + C + D



###############################################################################
###############################################################################
###############################################################################  


#verification later...
'''
A_realvalues = data.A[:7].values
B_realvalues = data.B[:7].values
C_realvalues = data.C[:7].values
D_realvalues = data.D[:7].values
'''

###############################################################################
###############################################################################
###############################################################################  


#make figure layout
fig = plt.figure(figsize=(8.,4.))

#set axes for the plot within the figure
ax = fig.add_axes([0.0, 0., 1., 1.]) 

ls1, = ax.plot(time_values, Ao_realvalues, c='grey', zorder=1)

for i in range(0,number_of_samples):

    print i
    
    #the line plots, time versus data
    ls2, = ax.plot(time_values, A_solutions[i,:], c='goldenrod', zorder=2)
    ls3, = ax.plot(time_values, B_solutions[i,:], c='chocolate', zorder=3)
    ls4, = ax.plot(time_values, C_solutions[i,:], c='royalblue', zorder=4)
    ls5, = ax.plot(time_values, D_solutions[i,:], c='dodgerblue', zorder=5)    

    #the scatter points of data observations
    ax.scatter(time_values, A_solutions[i,:], c='goldenrod', zorder=2)
    ax.scatter(time_values, B_solutions[i,:], c='chocolate', zorder=3)
    ax.scatter(time_values, C_solutions[i,:], c='royalblue', zorder=4)
    ax.scatter(time_values, D_solutions[i,:], c='dodgerblue', zorder=5)

#force plot to not have any buffer space
plt.margins(x=0, y=0)


ax.set_xlabel(r'Time (s; $ \times 10^{5}$)', fontsize=12, color='k')
ax.set_ylabel(r'[Mo] ($\mu$m; $ \times 10^{-5}$)', fontsize=12, color='k')

ax.legend([ls1, ls2, ls3, ls4, ls5],
          ['MoO$_{4}$','MoO$_{3}$S','MoOS$_{3}$','MoO$_{2}$S$_{2}$','MoS$_{4}$'],
                              loc='upper right',shadow=True, 
          fancybox=True, ncol=2, fontsize=12, framealpha=1.)

ax.set_xticks([0.,100000,200000,300000,400000])
ax.set_xticklabels(['0','1','2','3','4'], fontsize=12)

ax.set_yticks([0.,0.00001,0.00002,0.00003,0.00004,0.00005])
ax.set_yticklabels(['0','1','2','3','4','5'], fontsize=12)

ax.grid(which='major', axis='both', linestyle='--', alpha=0.5)

#save your image
plt.savefig('image_stephA.png', bbox_inches='tight', pad_inches=0.075, dpi=200)
plt.close()


###############################################################################
###############################################################################
###############################################################################  



  
