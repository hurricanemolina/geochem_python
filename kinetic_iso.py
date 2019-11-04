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


number_of_samples = 10


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
data = xr.open_dataset('file1_new.nc', decode_cf=True)
data2 = xr.open_dataset('file2_new.nc', decode_cf=True)

#extract data variables and limit all the data to the first 7-time points.
#time_values = data.time[:7].values

#make new normally spaced time steps to see if the graph will work at longer time_values
def range_inc(start, stop, step, div):
    i = start
    calc = [0]
    while i < stop:
        calc.append((data.time[len(data.time)-1].values / ((stop - i) * div)))
        i += step
    return calc

time_values = range_inc(0,1000,1,10)

#Ao_realvalues = data.A_o[:9].values

###############################################################################
###############################################################################
###############################################################################


#Create functions to be called to compute the different chemical parameters

def Mo_solver(Sig_S, eq, gtlt):
    A = A_o * np.exp(-ki*t)
         #K_a = (A * DEL_S) / ((A_o-A) * S_tot)
         #isoA = (K_a - 1) * 1000
    DEL_S = Sig_S
    B = np.divide((ki*A_o),(kii-ki)) * (np.exp(-ki*t)-np.exp(-kii*t))
    K_b = (B*DEL_S) / (A * S_tot)
    isoB = (K_b - 1) * 1000

    C = (np.divide((kii*ki*A_o),(kii-ki)) * \
        ((np.divide(1,(kiii-ki)) * np.exp(-ki*t)) - \
        (np.divide(1,(kiii-kii)) * (np.exp(-kii*t))) + \
        (np.divide(1 ,(kiii-kii)) * (np.exp(-kiii*t))) - \
        (np.divide(1 ,(kiii-ki)) * (np.exp(-kiii*t)))))
    #K_c = (C*DEL_S) / (B * S_tot)
    #isoC = (K_c - 1) * 1000
    #DEL_S = Sig_S - (B + C)

    D = np.divide((kiii*kii*ki*A_o), (kii-ki))  * \
        ((np.divide(1, ((kiii-ki)*(kiiii-ki))) * np.exp(-ki*t)) - \
        (np.divide(1, ((kiii-kii)*(kiiii-kii))) * np.exp(-kii*t)) + \
        (np.divide(1, ((kiii-kii)*(kiiii-kiii))) * np.exp(-kiii*t)) - \
        (np.divide(1, ((kiii-ki)*(kiiii-kiii))) * np.exp(-kiii*t)) - \
        (np.divide(1, ((kiii-ki)*(kiiii-ki))) * np.exp(-kiiii*t)) + \
        (np.divide(1, ((kiii-kii)*(kiiii-kii))) * np.exp(-kiiii*t)) - \
        (np.divide(1, ((kiii-kii)*(kiiii-kiii))) * np.exp(-kiiii*t)) + \
        (np.divide(1, ((kiii-ki)*(kiiii-kiii))) * np.exp(-kiiii*t)))
    #K_d = (D *DEL_S) / (C )
    #isoD = (K_d - 1) * 100
    #DEL_S = Sig_S - (B + C + D)

    #E = (A_o - A - B - C - D)
    E = A_o + (np.divide((kiiii*kiii*kii*ki*A_o), (kii-ki)) * \
        (np.divide(np.exp(-ki*t), ((kiii-ki)*(kiiii-ki)*-ki)) - \
        np.divide(np.exp(-kii*t), ((kiii-kii)*(kiiii-kii)*-kii)) + \
        np.divide(np.exp(-kiii*t), ((kiii-kii)*(kiiii-kiii)*-kiii)) - \
        np.divide(np.exp(-kiii*t), ((kiii-ki)*(kiiii-kiii)*-kiii)) - \
        np.divide(np.exp(-kiiii*t), ((kiii-ki)*(kiiii-ki)*-kiiii)) + \
        np.divide(np.exp(-kiiii*t), ((kiii-kii)*(kiiii-kii)*-kiiii)) - \
        np.divide(np.exp(-kiiii*t), ((kiii-kii)*(kiiii-kiii)*-kiiii)) + \
        np.divide(np.exp(-kiiii*t), ((kiii-ki)*(kiiii-kiii)*-kiiii))))
    #DEL_S = Sig_S - (B + C + D + E)

    #K_e = K_d
    #isoE = (K_e - 1) * 1000
    if eq == 1:
        return A
    if eq == 2:
        return B
    if eq == 3:
        return C
    if eq == 4:
        return D
    else:
        return E


###############################################################################
###############################################################################
###############################################################################


#extracting randomized k-values for 10,000 member randomization

#we are setting a seed for randomization for testing and ensuring we get same random results every time.
#we are bootstrapping, allowing for values to get called more than once with numpy's random choice

#get input from user to sulfide values
User_Mo = float(input('estimated total Mo conc.?  '))
User_Sulfide = float(input('estimated sulfide value?  '))
User_pH = int(input('estimated pH value?  '))

Ao_realvalues = []
for z in time_values:
    Ao_realvalues.append(User_Mo)

if User_pH < 7:
    ki_range = ki_range * 2
    kii_range = kii_range * 2
    kiii_range = kiii_range * 2
    kiiii_range = kiiii_range * 2

if User_pH > 7:
    ki_range = ki_range / 2
    kii_range = kii_range / 2
    kiii_range = kiii_range / 2
    kiiii_range = kiiii_range / 2

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
A_solutions = np.zeros((number_of_samples, len(time_values)))
B_solutions = np.zeros((number_of_samples, len(time_values)))
C_solutions = np.zeros((number_of_samples, len(time_values)))
D_solutions = np.zeros((number_of_samples, len(time_values)))
E_solutions = np.zeros((number_of_samples, len(time_values)))

R = 8.314
T = 298
DEL_G = -33
DEL_sigS = 0
S_tot = 5e-5 * 10

#add noise to the sulfide values
Sulfide_low = User_Sulfide / 1.1
Sulfide_high = User_Sulfide * 1.1
Rand_Sulfide = np.array([np.random.uniform(Sulfide_low, Sulfide_high) for i in range(number_of_samples)])
#np.arange(Sulfide_low, Sulfide_high)

checker = True
sul_ck = True
#DEL_sigS = np.array(np.nan() for i in range(number_of_samples))
B_switch = 1
C_switch = 1
D_switch = 1
E_switch = 1


for indexer, (ki, kii, kiii, kiiii) in enumerate(zip(ki_randomized, kii_randomized, kiii_randomized, kiiii_randomized)):

    for sec_indexer, (t, A_o) in enumerate(zip(time_values, Ao_realvalues)):

        A_solutions[indexer, sec_indexer] = Mo_solver(Rand_Sulfide[indexer], 1, sul_ck)
        B_solutions[indexer, sec_indexer] = Mo_solver(Rand_Sulfide[indexer], 2, sul_ck)
        C_solutions[indexer, sec_indexer] = Mo_solver(Rand_Sulfide[indexer], 3, sul_ck)
        D_solutions[indexer, sec_indexer] = Mo_solver(Rand_Sulfide[indexer], 4, sul_ck)
        E_solutions[indexer, sec_indexer] = Mo_solver(Rand_Sulfide[indexer], 5, sul_ck)

###############################################################################
###############################################################################
###############################################################################

#repeat the loops to calculate sulide consumed, truncate data accordingly

Species_cutoff = np.zeros((number_of_samples, len(time_values)))

#Check if the species is decreasing, don't double count moles of sulfide consumed

for Sindexer, Svalue in enumerate(Rand_Sulfide):
    #find the index of the max value for each species
    B_idxmax = B_solutions[Sindexer].argmax()
    C_idxmax = C_solutions[Sindexer].argmax()
    D_idxmax = D_solutions[Sindexer].argmax()
    E_idxmax = E_solutions[Sindexer].argmax()
    #calculate the end point of sulfide
    i=0
    checker = True
    while i < B_idxmax and checker == True:
        DEL_sigS = Rand_Sulfide[Sindexer] - B_solutions[Sindexer][i] - C_solutions[Sindexer][i] - D_solutions[Sindexer][i] - E_solutions[Sindexer][i]
        if DEL_sigS <= 0:
            print('B species', i, Sindexer)
            #Species_cutoff[Sindexer, i] = 0
            checker = False
        i += 1
    i = 0
    while i < C_idxmax and checker == True:
        DEL_sigS = Rand_Sulfide[Sindexer] - B_solutions[Sindexer][i] - C_solutions[Sindexer][i] - D_solutions[Sindexer][i] - E_solutions[Sindexer][i]
        if DEL_sigS <= 0:
            print('C_species', i, Sindexer)
            checker = False
        i+=1
    i = 0
    while i < D_idxmax and checker == True:
        DEL_sigS = Rand_Sulfide[Sindexer] - B_solutions[Sindexer][i] - C_solutions[Sindexer][i] - D_solutions[Sindexer][i] - E_solutions[Sindexer][i]
        if DEL_sigS <= 0:
            print('D_species', i, Sindexer)
            checker = False
        i+=1
    i = 0
    while i < D_idxmax and checker == True:
        DEL_sigS = Rand_Sulfide[Sindexer] - B_solutions[Sindexer][i] - C_solutions[Sindexer][i] - D_solutions[Sindexer][i] - E_solutions[Sindexer][i]
        if DEL_sigS <= 0:
            print('E_species', i, Sindexer)
            checker = False
        i += 1
    if checker == True:
        print('all_species', Sindexer)


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

#ls1, = ax.plot(time_values, Ao_realvalues, c='black', zorder=6)

for i in range(0,number_of_samples):

    #print i

    #the line plots, time versus data
    ls2, = ax.plot(time_values, A_solutions[i,:], c='grey', zorder=5, alpha=0.3)
    ls3, = ax.plot(time_values, B_solutions[i,:], c='goldenrod', zorder=4, alpha=0.3)
    ls4, = ax.plot(time_values, C_solutions[i,:], c='chocolate', zorder=3, alpha=0.3)
    ls5, = ax.plot(time_values, D_solutions[i,:], c='royalblue', zorder=2, alpha=0.3)
    ls6, = ax.plot(time_values, E_solutions[i,:], c='dodgerblue', zorder=1, alpha=0.3)
'''
    #the scatter points of data observations
    ax.scatter(time_values, A_solutions[i,:], c='grey', zorder=5, alpha=0.1)
    ax.scatter(time_values, B_solutions[i,:], c='goldenrod', zorder=4, alpha=0.3)
    ax.scatter(time_values, C_solutions[i,:], c='chocolate', zorder=3, alpha=0.3)
    ax.scatter(time_values, D_solutions[i,:], c='royalblue', zorder=2, alpha=0.3)
    ax.scatter(time_values, E_solutions[i,:], c='dodgerblue', zorder=1,  alpha=0.3)
'''
#force plot to not have any buffer space
plt.margins(x=0, y=0)

ax.set_xlabel(r'Time (s; $ \times 10^{5}$)', fontsize=12, color='k')
ax.set_ylabel(r'[Mo] ($\mu$m; $ \times 10^{-5}$)', fontsize=12, color='k')

line1 = pylab.Line2D(range(10),range(10),marker="_",linewidth=2.0,color="grey",alpha=1.0)
line2 = pylab.Line2D(range(10),range(10),marker="_",linewidth=2.0,color="goldenrod",alpha=1.0)
line3 = pylab.Line2D(range(10),range(10),marker="_",linewidth=2.0,color="chocolate",alpha=1.0)
line4 = pylab.Line2D(range(10),range(10),marker="_",linewidth=2.0,color="royalblue",alpha=1.0)
line5 = pylab.Line2D(range(10),range(10),marker="_",linewidth=2.0,color="dodgerblue",alpha=1.0)

ax.legend([line1, line2, line3, line4, line5],
          ['MoO$_{4}$','MoO$_{3}$S','MoO$_{2}$S$_{2}$','MoOS$_{3}$', 'MoS$_{4}$'],
                              loc='upper right',
          fancybox=True, ncol=2, fontsize=12, framealpha=0.9)

ax.set_xticks([0.,100000,200000,300000,400000,500000,600000,700000,800000,900000,1000000, \
1100000,1200000,1300000,1400000,1500000,1600000,1700000,1800000,1900000,2000000])
ax.set_xticklabels(['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'], fontsize=12)

ax.set_yticks([0.,0.00001,0.00002,0.00003,0.00004,0.00005])
ax.set_yticklabels(['0','1','2','3','4','5'], fontsize=12)

#ax.grid(which='major', axis='both', linestyle='--', alpha=0.5)

plt.savefig('image_iso.png', bbox_inches='tight', pad_inches=0.075, dpi=200, alpha=0.004)
plt.show()
plt.close()

time_hours = time_values
#save your image
#plt.plot(time_values, A_solutions[0], c='grey', zorder=5, alpha=0.3)
plt.plot(time_hours[:50], -np.log(B_solutions[0][:50]), c='goldenrod', zorder=4, alpha=1)
#plt.plot(time_hours[:50], np.log(C_solutions[0][:50]), c='chocolate', zorder=3, alpha=1)
plt.plot(time_hours[:50], -np.log(D_solutions[0][:50]), c='royalblue', zorder=2, alpha=1)
#plt.plot(time_hours[50:], np.log(E_solutions[0][50:]), c='dodgerblue', zorder=1, alpha=1)


plt.savefig('ln_C.png', bbox_inches='tight', pad_inches=0.075, dpi=200, alpha=0.004)
plt.show()
plt.close()


###############################################################################
###############################################################################
###############################################################################
