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
import pylab


###############################################################################
###############################################################################
###############################################################################


number_of_samples = 100


###############################################################################
###############################################################################
###############################################################################


#k value ranges

#MARIA'S QUESTION: do we no longer need kiiii? noticed we don't have this variable in equations.

#k values are based on experiments in: Clarke et al (1987) Kinetics of the formation and hydrolysis
#reaction of some thiomolybdate(VI) anions in aqueous solution. Inorg. Chim. Acta
#and Harmer & Sykes (1980). Kinetics of the Interconversion of Sulfido- and
#Oxomolybdate(VI) Species MoOxS4-x2- in Aqueous Solutions. Inorganic Chemistry

ki_low = 9e-5 #k_{01}
kii_low = 9e-6 #k_{12}
kiii_low = 9e-6 #k_{23}
kiiii_low = 2e-5 #K_{34}

ki_hi = 2e-4
kii_hi = 2e-5
kiii_hi = 2e-5
kiiii_hi = 5e-5



ki_range = np.linspace(ki_low,ki_hi,10)
kii_range = np.linspace(kii_low,kii_hi,10)
kiii_range = np.linspace(kiii_low,kiii_hi,10)
kiiii_range = np.linspace(kiiii_low,kiiii_hi,10)

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
rand_ki = np.array([np.random.choice(10) for i in range(number_of_samples)])

np.random.seed(1)
rand_kii = np.array([np.random.choice(10) for i in range(number_of_samples)])

np.random.seed(2)
rand_kiii = np.array([np.random.choice(10) for i in range(number_of_samples)])

np.random.seed(3)
rand_kiiii = np.array([np.random.choice(10) for i in range(number_of_samples)])


ki_randomized = ki_range[rand_ki]
kii_randomized = kii_range[rand_kii]
kiii_randomized = kiii_range[rand_kiii]
kiiii_randomized = kiiii_range[rand_kiiii]


###############################################################################
###############################################################################
###############################################################################


#creating blank arrays to fill.
A_solutions = np.zeros((number_of_samples,7))
B_solutions = np.zeros((number_of_samples,7))
C_solutions = np.zeros((number_of_samples,7))
D_solutions = np.zeros((number_of_samples,7))
E_solutions = np.zeros((number_of_samples,7))

for indexer, (ki, kii, kiii, kiiii) in enumerate(zip(ki_randomized, kii_randomized, kiii_randomized, kiiii_randomized)):

    for sec_indexer, (t, A_o) in enumerate(zip(time_values, Ao_realvalues)):

        # we will only have 4 equations since we are ignoring back reations, so the reactions proceed from A->D

        A = A_o * np.exp(-ki*t)


        B = np.divide((ki*A_o),(kii-ki)) * (np.exp(-ki*t)-np.exp(-kii*t))

        #Subject to change, Natalia and Dimtry will check my math for these equations

        C = (np.divide((kii*ki*A_o),(kii-ki)) * \
                    ((np.divide(1,(kiii-ki)) * np.exp(-ki*t)) - \
                    (np.divide(1,(kiii-kii)) * (np.exp(-kii*t))) + \
                    (np.divide(1 ,(kiii-kii)) * (np.exp(-kiii*t))) - \
                    (np.divide(1 ,(kiii-ki)) * (np.exp(-kiii*t)))))
        #MARIA'S COMMENT: Please double check bracket placements in equation for D! -- SRH they are correct

        D = np.divide((kiii*kii*ki*A_o), (kii-ki))  * \
                      ((np.divide(1, ((kiii-ki)*(kiiii-ki))) * np.exp(-ki*t)) - \
                       (np.divide(1, ((kiii-kii)*(kiiii-kii))) * np.exp(-kii*t)) + \
                       (np.divide(1, ((kiii-kii)*(kiiii-kiii))) * np.exp(-kiii*t)) - \
                       (np.divide(1, ((kiii-ki)*(kiiii-kiii))) * np.exp(-kiii*t)) - \
                       (np.divide(1, ((kiii-ki)*(kiiii-ki))) * np.exp(-kiiii*t)) + \
                       (np.divide(1, ((kiii-kii)*(kiiii-kii))) * np.exp(-kiiii*t)) - \
                       (np.divide(1, ((kiii-kii)*(kiiii-kiii))) * np.exp(-kiiii*t)) + \
                       (np.divide(1, ((kiii-ki)*(kiiii-kiii))) * np.exp(-kiiii*t)))

        E = A_o - A - B - C - D

        '''
        E = np.divide((kiiii*kiii*kii*ki*A_o), (kii-ki))  * \
                      ((np.divide(1, ((kiii-ki)*(kiiii-ki)*(-ki))) * np.exp(-ki*t)) - \
                       (np.divide(1, ((kiii-kii)*(kiiii-kii)*(-kii))) * np.exp(-kii*t)) + \
                       (np.divide(1, ((kiii-kii)*(kiiii-kiii)*(-kiii))) * np.exp(-kiii*t)) - \
                       (np.divide(1, ((kiii-ki)*(kiiii-kiii)*(-kiii))) * np.exp(-kiii*t)) - \
                       (np.divide(1, ((kiii-ki)*(kiiii-ki)*(-kiiii))) * np.exp(-kiiii*t)) + \
                       (np.divide(1, ((kiii-kii)*(kiiii-kii)*(-kiiii))) * np.exp(-kiiii*t)) - \
                       (np.divide(1, ((kiii-kii)*(kiiii-kiii)*(-kiiii))) * np.exp(-kiiii*t)) + \
                       (np.divide(1, ((kiii-ki)*(kiiii-kiii)*(-kiiii))) * np.exp(-kiiii*t)) - \
                       np.divide(1, ((kiii-ki)*(kiiii-ki)*(-ki))) + \
                       np.divide(1, ((kiii-kii)*(kiiii-kii)*(-kii))) - \
                       np.divide(1, ((kiii-kii)*(kiiii-kiii)*(-kiii))) + \
                       np.divide(1, ((kiii-ki)*(kiiii-kiii)*(-kiii))) + \
                       np.divide(1, ((kiii-ki)*(kiiii-ki)*(-kiiii))) - \
                       np.divide(1, ((kiii-kii)*(kiiii-kii)*(-kiiii))) + \
                       np.divide(1, ((kiii-kii)*(kiiii-kiii)*(-kiiii))) - \
                       np.divide(1, ((kiii-kii)*(kiiii-kiii)*(-kiiii))))
        '''

        A_solutions[indexer, sec_indexer] = A
        B_solutions[indexer, sec_indexer] = B
        C_solutions[indexer, sec_indexer] = C
        D_solutions[indexer, sec_indexer] = D
        E_solutions[indexer, sec_indexer] = E


    #print indexer


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

ls1, = ax.plot(time_values, Ao_realvalues, c='black', zorder=6)

for i in range(0,number_of_samples):

    #print i

    #the line plots, time versus data
    ls2, = ax.plot(time_values, A_solutions[i,:], c='grey', zorder=5, alpha=0.3)
    ls3, = ax.plot(time_values, B_solutions[i,:], c='goldenrod', zorder=4, alpha=0.3)
    ls4, = ax.plot(time_values, C_solutions[i,:], c='chocolate', zorder=3, alpha=0.3)
    ls5, = ax.plot(time_values, D_solutions[i,:], c='royalblue', zorder=2, alpha=0.3)
    ls6, = ax.plot(time_values, E_solutions[i,:], c='dodgerblue', zorder=1, alpha=0.3)

    #the scatter points of data observations
    ax.scatter(time_values, A_solutions[i,:], c='grey', zorder=5, alpha=0.1)
    ax.scatter(time_values, B_solutions[i,:], c='goldenrod', zorder=4, alpha=0.3)
    ax.scatter(time_values, C_solutions[i,:], c='chocolate', zorder=3, alpha=0.3)
    ax.scatter(time_values, D_solutions[i,:], c='royalblue', zorder=2, alpha=0.3)
    ax.scatter(time_values, E_solutions[i,:], c='dodgerblue', zorder=1,  alpha=0.3)

#force plot to not have any buffer space
plt.margins(x=0, y=0)


ax.set_xlabel(r'Time (s; $ \times 10^{5}$)', fontsize=12, color='k')
ax.set_ylabel(r'[Mo] ($\mu$m; $ \times 10^{-5}$)', fontsize=12, color='k')

line1 = pylab.Line2D(range(10),range(10),marker="_",linewidth=2.0,color="grey",alpha=1.0)
line2 = pylab.Line2D(range(10),range(10),marker="_",linewidth=2.0,color="goldenrod",alpha=1.0)
line3 = pylab.Line2D(range(10),range(10),marker="_",linewidth=2.0,color="chocolate",alpha=1.0)
line4 = pylab.Line2D(range(10),range(10),marker="_",linewidth=2.0,color="royalblue",alpha=1.0)
line5 = pylab.Line2D(range(10),range(10),marker="_",linewidth=2.0,color="dodgerblue",alpha=1.0)

ax.legend([ls1, line1, line2, line3, line4, line5],
          ['Mo$_{tot}$','MoO$_{4}$','MoO$_{3}$S','MoO$_{2}$S$_{2}$','MoOS$_{3}$', 'MoS$_{4}$'],
                              loc='upper right',
          fancybox=True, ncol=2, fontsize=12, framealpha=0.9)

ax.set_xticks([0.,100000,200000,300000,400000])
ax.set_xticklabels(['0','1','2','3','4'], fontsize=12)

ax.set_yticks([0.,0.00001,0.00002,0.00003,0.00004,0.00005])
ax.set_yticklabels(['0','1','2','3','4','5'], fontsize=12)

ax.grid(which='major', axis='both', linestyle='--', alpha=0.5)

#save your image
plt.savefig('image_stephA.png', bbox_inches='tight', pad_inches=0.075, dpi=200, alpha=0.004)
plt.close()


###############################################################################
###############################################################################
###############################################################################
