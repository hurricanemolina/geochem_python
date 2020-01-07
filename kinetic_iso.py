#!/usr/bin/env python3
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

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pylab


###############################################################################
###############################################################################
###############################################################################

#get the number of runs for the model
number_of_samples = int(input('number of iterations?  '))

#get input from user to sulfide values
User_Mo = float(input('estimated total Mo conc.?  '))
User_Sulfide = float(input('estimated sulfide value?  '))
User_pH = int(input('estimated pH value?  '))


###############################################################################
###############################################################################
###############################################################################


#k value ranges

#MARIA'S QUESTION: do we no longer need kiiii? noticed we don't have this variable in equations.

#k values are based on experiments in: Clarke et al (1987) Kinetics of the formation and hydrolysis
#reaction of some thiomolybdate(VI) anions in aqueous solution. Inorg. Chim. Acta
#and Harmer & Sykes (1980). Kinetics of the Interconversion of Sulfido- and
#Oxomolybdate(VI) Species MoOxS4-x2- in Aqueous Solutions. Inorganic Chemistry

'''
ki_low = 1e-4 #k_{01}
kii_low = 9e-5 #k_{12}
kiii_low = 1e-5 #k_{23}
kiiii_low = 1e-6 #K_{34}

ki_hi = 2e-4
kii_hi = 2e-4
kiii_hi = 3e-5
kiiii_hi = 5e-6
'''

ki_low = 1e-4 #k_{01}
kii_low = 4e-6 #k_{12}
kiii_low = 3e-6 #k_{23} 9e-7
kiiii_low = 2e-6 #K_{34} 8e-7 #K_{34}

ki_hi = 3e-4
kii_hi = 6e-6
kiii_hi = 5e-6
kiiii_hi = 4e-6

ki_range = np.linspace(ki_low,ki_hi,10)
kii_range = np.linspace(kii_low,kii_hi,10)
kiii_range = np.linspace(kiii_low,kiii_hi,10)
kiiii_range = np.linspace(kiiii_low,kiiii_hi,10)

#Fractionation factor calculated from expertimental data
del_A = 0.05
del_B = 0.59
del_C = -0.17
del_D = -0.15
del_E = -0.32

'''
#calculate the influence of pH on the reaction rates
ki_range =  ki_range * 2
kii_range = kii_range * 2
kiii_range = 1000000000 * User_pH**-14.61
kiiii_range = 4508 * User_pH**-9.93
'''

#calculate the influence of sulfide on rate reations (Clarke and Laurie, 1987)
sulfide_Mo_ratio = User_Sulfide / User_Mo
ki_range =  ki_range * (0.427**sulfide_Mo_ratio / ki_range)
kii_range = ki_range * (0.2999**sulfide_Mo_ratio / ki_range)
kiii_range = kiii_range * sulfide_Mo_ratio
kiiii_range = kiiii_range * sulfide_Mo_ratio

###############################################################################
###############################################################################
###############################################################################

#time_values = range_inc(0,1000,1,10)
time_values = np.linspace(0, 500000, 1000)[0:]

###############################################################################
###############################################################################
###############################################################################

#creat artificial boundry for minimum concentration and maximum concentration
A_min = User_Mo * 0.02
A_max = User_Mo
B_min = User_Mo * 0.02
B_max = User_Mo
C_min = User_Mo * 0.04
C_max = User_Mo
D_min = User_Mo * 0.02
D_max = User_Mo
E_min = User_Mo * 0.02
E_max = User_Mo

###############################################################################
###############################################################################
###############################################################################

#Create functions to be called to compute the different chemical parameters

def Mo_solver(Sig_S, eq, divtime, min, max):
    A = A_o * np.exp(-ki*divtime) + min

    B = np.divide((ki*A_o),(kii-ki)) * (np.exp(-ki*divtime)-np.exp(-kii*divtime))

    C = (np.divide((kii*ki*A_o),(kii-ki)) * \
        ((np.divide(1,(kiii-ki)) * np.exp(-ki*divtime)) - \
        (np.divide(1,(kiii-kii)) * (np.exp(-kii*divtime))) + \
        (np.divide(1 ,(kiii-kii)) * (np.exp(-kiii*divtime))) - \
        (np.divide(1 ,(kiii-ki)) * (np.exp(-kiii*divtime)))))

    D = min + (np.divide((kiii*kii*ki*A_o), (kii-ki))  * \
        ((np.divide(1, ((kiii-ki)*(kiiii-ki))) * np.exp(-ki*divtime)) - \
        (np.divide(1, ((kiii-kii)*(kiiii-kii))) * np.exp(-kii*divtime)) + \
        (np.divide(1, ((kiii-kii)*(kiiii-kiii))) * np.exp(-kiii*divtime)) - \
        (np.divide(1, ((kiii-ki)*(kiiii-kiii))) * np.exp(-kiii*divtime)) - \
        (np.divide(1, ((kiii-ki)*(kiiii-ki))) * np.exp(-kiiii*divtime)) + \
        (np.divide(1, ((kiii-kii)*(kiiii-kii))) * np.exp(-kiiii*divtime)) - \
        (np.divide(1, ((kiii-kii)*(kiiii-kiii))) * np.exp(-kiiii*divtime)) + \
        (np.divide(1, ((kiii-ki)*(kiiii-kiii))) * np.exp(-kiiii*divtime))))

    E = min + A_o + (np.divide((kiiii*kiii*kii*ki*A_o), (kii-ki)) * \
        (np.divide(np.exp(-ki*divtime), ((kiii-ki)*(kiiii-ki)*-ki)) - \
        np.divide(np.exp(-kii*divtime), ((kiii-kii)*(kiiii-kii)*-kii)) + \
        np.divide(np.exp(-kiii*divtime), ((kiii-kii)*(kiiii-kiii)*-kiii)) - \
        np.divide(np.exp(-kiii*divtime), ((kiii-ki)*(kiiii-kiii)*-kiii)) - \
        np.divide(np.exp(-kiiii*divtime), ((kiii-ki)*(kiiii-ki)*-kiiii)) + \
        np.divide(np.exp(-kiiii*divtime), ((kiii-kii)*(kiiii-kii)*-kiiii)) - \
        np.divide(np.exp(-kiiii*divtime), ((kiii-kii)*(kiiii-kiii)*-kiiii)) + \
        np.divide(np.exp(-kiiii*divtime), ((kiii-ki)*(kiiii-kiii)*-kiiii))))

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

#Calculate the isotopic ratio using the contrubition of each species, equations
#derived from emperical relationship to our data (Hlohowskyj et al., 2020)

def Mo_iso_solver(species, divtime, mole_ratio):
    if species == 'A':
        calc = (-3e-13 * divtime**2 + 1e-7 * divtime + 0.0419) / mole_ratio
        return calc
    if species == 'B':
        calc = (-4e-12 * divtime**2 + 1e-06 * divtime + 0.6516) / mole_ratio
        #calc = (-1e-6 * divtime + 0.9711) / mole_ratio
        return calc
    if species == 'C':
        #calc = (divtime**2 - 1E-05 * divtime + 0.01) / mole_ratio
        calc = (-4e-12 * divtime**2 + 4e-06 * divtime - 0.7475) / mole_ratio
        #calc = (3e-6 * divtime - 0.7303) / mole_ratio
        return calc
    if species == 'D':
        calc = (2e-12 * divtime**2 - 2e-6 * divtime - 0.042) / mole_ratio
        return calc
    if species == 'E':
        calc = (4e-13 * divtime**2 - 1e-06 * divtime - 0.0172) / mole_ratio
        return calc
    else:
        return

###############################################################################
###############################################################################
###############################################################################

#calculate the mole fraction of each species
def Mo_mole_frac(input):
    Mo_calc = input/User_Mo
    return Mo_calc

###############################################################################
###############################################################################
###############################################################################

#extracting randomized k-values for 10,000 member randomization

#we are setting a seed for randomization for testing and ensuring we get same random results every time.
#we are bootstrapping, allowing for values to get called more than once with numpy's random choice

Ao_realvalues = []
for z in time_values:
    Ao_realvalues.append(User_Mo)

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


#creating blank arrays to fill in
A_solutions = np.zeros((number_of_samples, len(time_values)))
B_solutions = np.zeros((number_of_samples, len(time_values)))
C_solutions = np.zeros((number_of_samples, len(time_values)))
D_solutions = np.zeros((number_of_samples, len(time_values)))
E_solutions = np.zeros((number_of_samples, len(time_values)))

#define chemical constants
R = 8.314
T = 298
DEL_G = -33
DEL_sigS = 0
S_tot = 5e-5 * 10

#add noise to the sulfide values
Sulfide_low = User_Sulfide / 1.1
Sulfide_high = User_Sulfide * 1.1

#set the distribution of possible sulfide values
Rand_Sulfide = np.array([np.random.uniform(Sulfide_low, Sulfide_high) for i in range(number_of_samples)])

checker = True
B_switch = 1
C_switch = 1
D_switch = 1
E_switch = 1

#Create loop to calculate each equation, using k values define above
for indexer, (ki, kii, kiii, kiiii) in enumerate(zip(ki_randomized, kii_randomized, kiii_randomized, kiiii_randomized)):

    for sec_indexer, (t, A_o) in enumerate(zip(time_values, Ao_realvalues)):

        A_solutions[indexer, sec_indexer] = Mo_solver(Rand_Sulfide[indexer], 1, time_values[sec_indexer], A_min, A_max)
        B_solutions[indexer, sec_indexer] = Mo_solver(Rand_Sulfide[indexer], 2, time_values[sec_indexer], B_min, B_max)
        C_solutions[indexer, sec_indexer] = Mo_solver(Rand_Sulfide[indexer], 3, time_values[sec_indexer], C_min, C_max)
        D_solutions[indexer, sec_indexer] = Mo_solver(Rand_Sulfide[indexer], 4, time_values[sec_indexer], D_min, D_max)
        E_solutions[indexer, sec_indexer] = Mo_solver(Rand_Sulfide[indexer], 5, time_values[sec_indexer], E_min, E_max)


###############################################################################
###############################################################################
###############################################################################

#repeat the loops to calculate sulide consumed, truncate data accordingly
Species_cutoff = np.zeros((number_of_samples, len(time_values)))

dominant_species = np.zeros(shape=(5,1))

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
            dominant_species[0] +=  1
            #Species_cutoff[Sindexer, i] = 0
            checker = False
        i += 1
    i = 0

    while i < C_idxmax and checker == True:
        DEL_sigS = Rand_Sulfide[Sindexer] - B_solutions[Sindexer][i] - C_solutions[Sindexer][i] - D_solutions[Sindexer][i] - E_solutions[Sindexer][i]
        if DEL_sigS <= 0:
            dominant_species[1] += 1
            checker = False
        i+=1
    i = 0
    while i < D_idxmax and checker == True:
        DEL_sigS = Rand_Sulfide[Sindexer] - B_solutions[Sindexer][i] - C_solutions[Sindexer][i] - D_solutions[Sindexer][i] - E_solutions[Sindexer][i]
        if DEL_sigS <= 0:
            dominant_species[2] += 1
            checker = False
        i+=1
    i = 0

    while i < D_idxmax and checker == True:
        DEL_sigS = Rand_Sulfide[Sindexer] - B_solutions[Sindexer][i] - C_solutions[Sindexer][i] - D_solutions[Sindexer][i] - E_solutions[Sindexer][i]
        if DEL_sigS <= 0:
            dominant_species[3] += 1
            checker = False
        i += 1

    if checker == True:
        dominant_species[4] += 1

print(dominant_species)

###############################################################################
###############################################################################
###############################################################################

#Create arrays to setup molar ratios

A_mole_ratio = np.zeros((number_of_samples, len(time_values)))
B_mole_ratio = np.zeros((number_of_samples, len(time_values)))
C_mole_ratio = np.zeros((number_of_samples, len(time_values)))
D_mole_ratio = np.zeros((number_of_samples, len(time_values)))
E_mole_ratio = np.zeros((number_of_samples, len(time_values)))

for indexer, (ki, kii, kiii, kiiii) in enumerate(zip(ki_randomized, kii_randomized, kiii_randomized, kiiii_randomized)):

    for sec_indexer, t in enumerate(time_values):

        A_mole_ratio[indexer, sec_indexer] = Mo_mole_frac(A_solutions[indexer, sec_indexer])
        B_mole_ratio[indexer, sec_indexer] = Mo_mole_frac(B_solutions[indexer, sec_indexer])
        C_mole_ratio[indexer, sec_indexer] = Mo_mole_frac(C_solutions[indexer, sec_indexer])
        D_mole_ratio[indexer, sec_indexer] = Mo_mole_frac(D_solutions[indexer, sec_indexer])
        E_mole_ratio[indexer, sec_indexer] = Mo_mole_frac(E_solutions[indexer, sec_indexer])

###############################################################################
###############################################################################
###############################################################################

#create lists for all isotope data

A_isotope_ratio = np.zeros((number_of_samples, len(time_values)))
B_isotope_ratio = np.zeros((number_of_samples, len(time_values)))
C_isotope_ratio = np.zeros((number_of_samples, len(time_values)))
D_isotope_ratio = np.zeros((number_of_samples, len(time_values)))
E_isotope_ratio = np.zeros((number_of_samples, len(time_values)))

###############################################################################
###############################################################################
###############################################################################

#calculate the isotopic value for each of the modeled species
#these constants are determined from emperical fits of isotope data

for indexer, ki in enumerate(ki_randomized):

    for sec_indexer, t in enumerate(time_values):

        A_isotope_ratio[indexer, sec_indexer] = Mo_iso_solver('A', time_values[sec_indexer], A_mole_ratio[indexer, sec_indexer])
        B_isotope_ratio[indexer, sec_indexer] = Mo_iso_solver('B', time_values[sec_indexer], B_mole_ratio[indexer, sec_indexer])
        C_isotope_ratio[indexer, sec_indexer] = Mo_iso_solver('C', time_values[sec_indexer], C_mole_ratio[indexer, sec_indexer])
        D_isotope_ratio[indexer, sec_indexer] = Mo_iso_solver('D', time_values[sec_indexer], D_mole_ratio[indexer, sec_indexer])
        E_isotope_ratio[indexer, sec_indexer] = Mo_iso_solver('E', time_values[sec_indexer], E_mole_ratio[indexer, sec_indexer])

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

    ls2, = ax.plot(time_values[:], A_solutions[i,:], c='grey', zorder=5, alpha=0.3)
    ls3, = ax.plot(time_values[:], B_solutions[i,:], c='goldenrod', zorder=4, alpha=0.3)
    ls4, = ax.plot(time_values[:], C_solutions[i,:], c='chocolate', zorder=3, alpha=0.3)
    ls5, = ax.plot(time_values[:], D_solutions[i,:], c='royalblue', zorder=2, alpha=0.3)
    ls6, = ax.plot(time_values[:], E_solutions[i,:], c='dodgerblue', zorder=1, alpha=0.3)

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

ax.legend([line1, line2, line3, line4, line5], \
         ['MoO$_{4}$','MoO$_{3}$S','MoO$_{2}$S$_{2}$','MoOS$_{3}$', 'MoS$_{4}$'], \
         loc='upper right', fancybox=True, ncol=2, fontsize=12, framealpha=0.9)

#ax.set_xticks([0.,100000,200000,300000,400000,500000,600000,700000,800000,900000,1000000, \
#1100000,1200000,1300000,1400000,1500000,1600000,1700000,1800000,1900000,2000000])
#ax.set_xticklabels(['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'], fontsize=12)

ax.set_xticks([0.,100000,200000,300000,400000,500000])
ax.set_xticklabels(['0','1','2','3','4','5'])

ax.set_yticks([0.,0.00001,0.00002,0.00003,0.00004,0.00005])
ax.set_yticklabels(['0','1','2','3','4','5'], fontsize=12)

ax.grid(which='major', axis='both', linestyle='--', alpha=0.5)
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
#plt.show()
plt.close()


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
    ls2, = ax.plot(time_values, A_mole_ratio[i,:], c='grey', zorder=5, alpha=0.3)
    ls3, = ax.plot(time_values, B_mole_ratio[i,:], c='goldenrod', zorder=4, alpha=0.3)
    ls4, = ax.plot(time_values, C_mole_ratio[i,:], c='chocolate', zorder=3, alpha=0.3)
    ls5, = ax.plot(time_values, D_mole_ratio[i,:], c='royalblue', zorder=2, alpha=0.3)
    ls6, = ax.plot(time_values, E_mole_ratio[i,:], c='dodgerblue', zorder=1, alpha=0.3)

#force plot to not have any buffer space
plt.margins(x=0, y=0)

ax.set_xlabel(r'Time (s; $ \times 10^{5}$)', fontsize=12, color='k')
ax.set_ylabel(r'[Mo] (percent)', fontsize=12, color='k')

line1 = pylab.Line2D(range(10),range(10),marker="_",linewidth=2.0,color="grey",alpha=1.0)
line2 = pylab.Line2D(range(10),range(10),marker="_",linewidth=2.0,color="goldenrod",alpha=1.0)
line3 = pylab.Line2D(range(10),range(10),marker="_",linewidth=2.0,color="chocolate",alpha=1.0)
line4 = pylab.Line2D(range(10),range(10),marker="_",linewidth=2.0,color="royalblue",alpha=1.0)
line5 = pylab.Line2D(range(10),range(10),marker="_",linewidth=2.0,color="dodgerblue",alpha=1.0)

ax.legend([line1, line2, line3, line4, line5],
          ['MoO$_{4}$','MoO$_{3}$S','MoO$_{2}$S$_{2}$','MoOS$_{3}$', 'MoS$_{4}$'],
                              loc='upper right',
          fancybox=True, ncol=2, fontsize=12, framealpha=0.9)

#ax.set_xticks([0.,1000000,2000000,3000000,4000000,5000000,6000000,7000000,8000000,9000000,10000000, \
#11000000,12000000,13000000,14000000,15000000,16000000,17000000,18000000,19000000,20000000])
#ax.set_xticklabels(['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'], fontsize=12)

ax.set_xticks([0.,100000,200000,300000,400000,500000])
ax.set_xticklabels(['0','1','2','3','4','5'])

ax.set_yticks([0.0,0.1,0.3,0.5,0.8,1])
ax.set_yticklabels(['0','10','30','50','80','100'], fontsize=12)

ax.grid(which='major', axis='both', linestyle='--', alpha=0.5)

plt.savefig('image_ratio.png', bbox_inches='tight', pad_inches=0.075, dpi=200, alpha=0.004)
#plt.show()
plt.close()


###############################################################################
###############################################################################
###############################################################################


plt.hist(dominant_species, bins=1)
plt.savefig('image_hist.png', bbox_inches='tight', pad_inches=0.075, dpi=200, alpha=0.004)
#plt.show()
plt.close()


###############################################################################
###############################################################################
###############################################################################

#make figure layout
fig = plt.figure(figsize=(8.,4.))

#set axes for the plot within the figure
ax = fig.add_axes([0.0, 0., 1., 1.])

#force plot to not have any buffer space
plt.margins(x=0, y=0)

ax.legend([line1, line2, line3, line4, line5],
          ['MoO$_{4}$','MoO$_{3}$S','MoO$_{2}$S$_{2}$','MoOS$_{3}$', 'MoS$_{4}$'],
                              loc='upper right',
          fancybox=True, ncol=2, fontsize=12, framealpha=0.9)

ax.set_xlabel(r'Time (s; $ \times 10^{5}$)', fontsize=12, color='k')
ax.set_ylabel(r'[$^{98}$Mo] (permille)', fontsize=12, color='k')

#define a horizontal line at 0 on y axis

ln_horizontal = np.zeros(len(time_values))

for i in range(0,number_of_samples):

    plt.plot(time_values, A_isotope_ratio[i], marker='', color='grey', alpha=0.3)
    plt.plot(time_values, B_isotope_ratio[i], marker='', color='goldenrod', alpha=0.3)
    plt.plot(time_values, C_isotope_ratio[i], marker='', color='chocolate', alpha=0.3)
    plt.plot(time_values, D_isotope_ratio[i], marker='', color='royalblue', alpha=0.3)
    plt.plot(time_values, E_isotope_ratio[i], marker='', color='dodgerblue', alpha=0.3)

plt.plot(time_values, ln_horizontal, marker='', color='black', linestyle='--', markeredgecolor='black', linewidth=0.5)


#plt.xlim(0,500000)

ax.set_xticks([0.,100000,200000,300000,400000,500000])
ax.set_xticklabels(['0','1','2','3','4','5'])

plt.ylim(-5,5)
plt.savefig('image_delta.png', bbox_inches='tight', pad_inches=0.075, dpi=200, alpha=0.004)
plt.show()
plt.close()


###############################################################################
###############################################################################
###############################################################################
'''
outfile = open('test.csv', 'w')
out = csv.writer(outfile, delimiter=',')
a = [14, 72, 173, 345, 518, 863]
#new = []
outfile.write('time, MoO4%, MoO3S%, MoO2S2%, MoOS3%, MoS4% \n')
for i in a:
    #new = [time_values[i], A_mole_ratio[0][i], B_mole_ratio[0][i], C_mole_ratio[0][i], D_mole_ratio[0][i], E_mole_ratio[0][i]]
    #out.writerows(zip(time_values[i], A_mole_ratio[0][i], B_mole_ratio[0][i], C_mole_ratio[0][i], D_mole_ratio[0][i], E_mole_ratio[0][i]))
    #new.append(zip(time_values[i], A_mole_ratio[0][i], B_mole_ratio[0][i], C_mole_ratio[0][i], D_mole_ratio[0][i], E_mole_ratio[0][i]))
    #new  = zip(time_values[i], A_mole_ratio[0][i], B_mole_ratio[0][i], C_mole_ratio[0][i], D_mole_ratio[0][i], E_mole_ratio[0][i])
    #for z in range(len(new)):
    #    outfile.write(f'{new[z]}\n')
    outfile.write(f'{time_values[i], A_mole_ratio[0][i], B_mole_ratio[0][i], C_mole_ratio[0][i], D_mole_ratio[0][i], E_mole_ratio[0][i]}\n')
outfile.close()

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def line_mean(array):
    #array = np.asarray(array)
    transp = array.T
    calc = []
    for i in range(len(transp)):
        calc.append(np.mean(transp[i]))
    return calc

plt.plot(time_values, A_solutions[0], marker='', color='grey', fillstyle='full', markeredgecolor='black',markeredgewidth=0.5)
plt.plot(time_values, test, marker='', color='red', fillstyle='full', markeredgecolor='blue',markeredgewidth=0.5)

'''
