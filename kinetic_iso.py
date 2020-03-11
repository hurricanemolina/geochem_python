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

#TODO list
#visual output of most probable species as histogram
# evelop a better determination of stochastic processes for model
#create interpolated data set from real data for machinelearning parameters

###############################################################################
###############################################################################
###############################################################################

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pylab
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


###############################################################################
###############################################################################
###############################################################################

#get the number of runs for the model
number_of_samples = int(input('number of iterations?  '))

#get input from user to sulfide values
User_Mo = float(input('estimated total Mo conc. (M)?  '))
User_Sulfide = float(input('estimated sulfide conc. (M)?  '))
User_pH = int(input('estimated pH value?  '))


###############################################################################
###############################################################################
###############################################################################

#based on experiments conducted at STARLAB in 2017
time_correlation = [0, 7200, 36000, 86400, 172800, 259200, 432000]

#import your data set in netcdf format
data = xr.open_dataset('file1_new.nc', decode_cf=True)

#extract data variables and limit all the data to the first 7-time points.
#time_values = data.time[:7].values

Ao_emperical = []
A_emperical = []
B_emperical = []
C_emperical = []
D_emperical = []
E_emperical = []

for z in range(len(time_correlation)):
    Ao_emperical.append(data.A_o[z].values)
    A_emperical.append(data.A[z].values)
    B_emperical.append(data.B[z].values)
    C_emperical.append(data.C[z].values)
    D_emperical.append(data.D[z].values)
    E_emperical.append(data.E[z].values)

data2 = xr.open_dataset('file2_new.nc', decode_cf=True)

Ao_iso_emperical = []
A_iso_emperical = []
B_iso_emperical = []
C_iso_emperical = []
D_iso_emperical = []
E_iso_emperical = []

for z in range(len(time_correlation)):
    Ao_iso_emperical.append(data2.A_o[z].values)
    A_iso_emperical.append(data2.A[z].values)
    B_iso_emperical.append(data2.B[z].values)
    C_iso_emperical.append(data2.C[z].values)
    D_iso_emperical.append(data2.D[z].values)
    E_iso_emperical.append(data2.E[z].values)



###############################################################################
###############################################################################
###############################################################################

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
#ki_range =  ki_range * (0.427**sulfide_Mo_ratio / ki_range)
#kii_range = ki_range * (0.2999**sulfide_Mo_ratio / ki_range)
#kiii_range = kiii_range * sulfide_Mo_ratio
#kiiii_range = kiiii_range * sulfide_Mo_ratio

###############################################################################
###############################################################################
###############################################################################

#time_values = range_inc(0,1000,1,10)
time_values = np.linspace(0, 500000, 1000)[0:]

###############################################################################
###############################################################################
###############################################################################

#creat artificial boundry for minimum concentration and maximum concentration
A_min = User_Mo * 0.01
A_max = User_Mo
B_min = User_Mo * 0.01
B_max = User_Mo
C_min = User_Mo * 0.01
C_max = User_Mo
D_min = User_Mo * 0.01
D_max = User_Mo
E_min = User_Mo * 0.01
E_max = User_Mo

###############################################################################
###############################################################################
###############################################################################

#simple function to find values for correlation fitness test

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

###############################################################################
###############################################################################
###############################################################################

#Create functions to be called to compute the different chemical parameters

def Mo_solver(Sig_S, eq, divtime, min, max):
    A = A_o * np.exp(-ki*divtime) + min

    B = np.divide((ki*A_o),(kii-ki)) * (np.exp(-ki*divtime)-np.exp(-kii*divtime)) + min

    C = min + (np.divide((kii*ki*A_o),(kii-ki)) * \
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
    A, B, C, D, E = [0,5], [0,5], [5,-5], [0,-5], [0,-5]
    if species == 'A':
        np.random.seed(0)
        #A_range = np.linspace(A[0],A[1],10)
        #A_iso_rand = np.array([np.random.choice(10) for i in range(number_of_samples)])
        #A_iso_test = A_range[A_iso_rand]
        #calc = (0.444 * np.log(divtime) -2.5127)
        #calc = (177827 * mole_ratio**3 - 17396 * mole_ratio**2 + 481.05 * mole_ratio - 1.0172)
        calc = -1.255 * np.log(mole_ratio) - 2.1249
        return calc
    if species == 'B':
        #calc = (-6e9 * mole_ratio**2 + 148945 * mole_ratio + 1.9459)
        #calc = (-14.773 * mole_ratio**2 + 7.6379 * mole_ratio + 1.9459)
        calc = (-1.603 * np.log(mole_ratio) + 0.4349)
        #calc = (-1e-6 * divtime + 0.9711) / mole_ratio
        #calc = (-4e-12 * divtime**2 + 1e-6 * divtime + 0.6516) / mole_ratio
        #calc = (8e-17 * divtime**3 -6e-11 * divtime**2 + 1e-5 * divtime + 0.4497) / mole_ratio
        return calc
    if species == 'C':
        #calc = (divtime**2 - 1E-05 * divtime + 0.01) / mole_ratio
        #calc = (-2e-12 * divtime**2 + 4e-06 * divtime - 0.0475) / mole_ratio
        calc = (3e-6 * divtime - 0.6933) / mole_ratio
        #calc = 149527 * mole_ratio - 2.944
        #calc = 2.9908 * np.log(mole_ratio) + 3.5551
        return calc
    if species == 'D':
        calc = (2e-12 * divtime**2 - 2e-6 * divtime - 0.042) / mole_ratio
        #calc = (-1e-8 * divtime - 2.1766) / mole_ratio
        #calc = 4e10 * mole_ratio**2 - 444940 * mole_ratio - 1.2036
        return calc
    if species == 'E':
        calc = (4e-13 * divtime**2 - 1e-06 * divtime - 0.0172) / mole_ratio
        #calc = (-1e-7 * divtime - 0.0274) / mole_ratio
        return calc
    else:
        return

from scipy import optimize

###############################################################################
###############################################################################
###############################################################################
'''
def isotope_mix(c):
    section = 0
    divtime = 300
    x_ratio = (A_solutions[section, divtime] / User_Mo, B_solutions[section, divtime] / User_Mo,
                C_solutions[section, divtime] / User_Mo, D_solutions[section, divtime] / User_Mo,
                E_solutions[section, divtime] / User_Mo)
    print(x_ratio)
    return ((c[0] * x_ratio[0]) + (c[1] * x_ratio[1]) +
            (c[2] * x_ratio[1]) + (c[3] * x_ratio[3]) +
            (c[4] * x_ratio[4]) - 0.17)

def iso_optimize_test():
    x0 = np.array([0]*5)
    x0_bound = (0,4)
    x1_bound = (0,5)
    x2_bound = (-4,5)
    x3_bound = (-5,0)
    x4_bound = (-4,0)

    iso_plot = []

    for i in range(0,5):
        section = i
        divtime = np.random.randint(0,500)
        result = optimize.minimize(isotope_mix, x0=x0,method='L-BFGS-B',
                           options={'maxiter':100})#, bounds=(x0_bound,x1_bound,x2_bound,x3_bound,x4_bound))

        iso_plot.append(result.x)
    return result, iso_plot
'''
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

#grab closest time value to emperical data collected in 2017
time_array = []

for i in range(len(time_correlation)):
    time_array.append(find_nearest(time_values, time_correlation[i]))

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


#add noise to the sulfide values
Sulfide_low = User_Sulfide / 1.1
Sulfide_high = User_Sulfide * 1.1

#set the distribution of possible sulfide values
Rand_Sulfide = np.array([np.random.uniform(Sulfide_low, Sulfide_high) for i in range(number_of_samples)])

checker = True

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

#verification of concentrations
A_model_corr = []
B_model_corr = []
C_model_corr = []
D_model_corr = []
E_model_corr = []

A_solutions_mean = A_solutions.mean(axis=0)
B_solutions_mean = B_solutions.mean(axis=0)
C_solutions_mean = C_solutions.mean(axis=0)
D_solutions_mean = D_solutions.mean(axis=0)
E_solutions_mean = E_solutions.mean(axis=0)

#create specific lists for testing correlation for each species
for i, q in time_array:
    A_model_corr.append(A_solutions_mean[i])
    B_model_corr.append(B_solutions_mean[i])
    C_model_corr.append(C_solutions_mean[i])
    D_model_corr.append(D_solutions_mean[i])
    E_model_corr.append(E_solutions_mean[i])

A_prime = np.array(A_model_corr)
A_prime = A_prime.reshape(-1,1)
B_prime = np.array(B_model_corr)
B_prime = B_prime.reshape(-1,1)
C_prime = np.array(C_model_corr)
C_prime = C_prime.reshape(-1,1)
D_prime = np.array(D_model_corr)
D_prime = D_prime.reshape(-1,1)
E_prime = np.array(E_model_corr)
E_prime = E_prime.reshape(-1,1)

plt.ylim(0)
plt.xlim(0)
plt.scatter(A_model_corr[:], A_emperical[:], c='grey', marker='o')
plt.scatter(B_model_corr[:], B_emperical[:], c='goldenrod', marker='o')
plt.scatter(C_model_corr[:], C_emperical[:], c='chocolate', marker='o')
plt.scatter(D_model_corr[:], D_emperical[:], c='royalblue', marker='o')
plt.scatter(E_model_corr[:], E_emperical[:], c='dodgerblue', marker='o')
plt.savefig('correlation.png', bbox_inches='tight', pad_inches=0.075, dpi=200)
#plt.show()
plt.close()

A_regression  = LinearRegression().fit(A_prime, A_emperical)
print('A model verus real conc. r^2', A_regression.score(A_prime, A_emperical))
B_regression  = LinearRegression().fit(B_prime, B_emperical)
print('B model verus real conc. r^2', B_regression.score(B_prime, B_emperical))
C_regression  = LinearRegression().fit(C_prime, C_emperical)
print('C model verus real conc. r^2', C_regression.score(C_prime, C_emperical))
D_regression  = LinearRegression().fit(D_prime, D_emperical)
print('D model verus real conc. r^2', D_regression.score(D_prime, D_emperical))
E_regression  = LinearRegression().fit(E_prime, E_emperical)
print('E model verus real conc. r^2', E_regression.score(E_prime, E_emperical))



###############################################################################
###############################################################################
###############################################################################

#verification of isotopic ratios
A_model_corr = []
B_model_corr = []
C_model_corr = []
D_model_corr = []
E_model_corr = []

A_iso_solutions_mean = A_isotope_ratio.mean(axis=0)
B_iso_solutions_mean = B_isotope_ratio.mean(axis=0)
C_iso_solutions_mean = C_isotope_ratio.mean(axis=0)
D_iso_solutions_mean = D_isotope_ratio.mean(axis=0)
E_iso_solutions_mean = E_isotope_ratio.mean(axis=0)

#create specific lists for testing correlation for each species
for i, q in time_array:
    A_model_corr.append(A_iso_solutions_mean[i])
    B_model_corr.append(B_iso_solutions_mean[i])
    C_model_corr.append(C_iso_solutions_mean[i])
    D_model_corr.append(D_iso_solutions_mean[i])
    E_model_corr.append(E_iso_solutions_mean[i])

A_prime = np.array(A_model_corr)
A_prime = A_prime.reshape(-1,1)
B_prime = np.array(B_model_corr)
B_prime = B_prime.reshape(-1,1)
C_prime = np.array(C_model_corr)
C_prime = C_prime.reshape(-1,1)
D_prime = np.array(D_model_corr)
D_prime = D_prime.reshape(-1,1)
E_prime = np.array(E_model_corr)
E_prime = E_prime.reshape(-1,1)

plt.scatter(A_model_corr[:], A_iso_emperical[:], c='grey', marker='o')
plt.scatter(B_model_corr[1:], B_iso_emperical[1:], c='goldenrod', marker='o')
plt.scatter(C_model_corr[:], C_iso_emperical[:], c='chocolate', marker='o')
plt.scatter(D_model_corr[:], D_iso_emperical[:], c='royalblue', marker='o')
plt.scatter(E_model_corr[:], E_iso_emperical[:], c='dodgerblue', marker='o')
plt.savefig('correlation_iso.png', bbox_inches='tight', pad_inches=0.075, dpi=200)
plt.ylim(0)
plt.xlim(0, np.max(A_model_corr))
#plt.show()
plt.close()

print('\n')
A_regression  = LinearRegression().fit(A_prime[1:], A_iso_emperical[1:])
print('A model verus real isotopes. r^2', A_regression.score(A_prime[1:], A_iso_emperical[1:]))
B_regression  = LinearRegression().fit(B_prime[1:], B_iso_emperical[1:])
print('B model verus real isotopes r^2', B_regression.score(B_prime[1:], B_iso_emperical[1:]))
C_regression  = LinearRegression().fit(C_prime[1:], C_iso_emperical[1:])
print('C model verus real isotopes. r^2', C_regression.score(C_prime[1:], C_iso_emperical[1:]))
D_regression  = LinearRegression().fit(D_prime[1:], D_iso_emperical[1:])
print('D model verus real isotopes r^2', D_regression.score(D_prime[1:], D_iso_emperical[1:]))
E_regression  = LinearRegression().fit(E_prime[1:], E_iso_emperical[1:])
print('E model verus real isotopes. r^2', E_regression.score(E_prime[1:], E_iso_emperical[1:]))

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
#plt.show()
plt.close()

time_hours = time_values / 3600


test1 = time_hours[50:].reshape(-1,1)
test2 = np.log10(B_solutions_mean[50:])
B_log_regression  = LinearRegression().fit(test1, test2)

plt.plot(time_hours[50:], np.log10(B_solutions_mean[50:]), c='goldenrod', zorder=1, alpha=1, marker='o')
plt.plot(test1, B_log_regression.predict(test1), c='black', zorder=2, alpha=1)
plt.xlim(0)
plt.savefig('ln_C_B.png', bbox_inches='tight', pad_inches=0.075, dpi=200, alpha=0.004)
plt.xlabel('Time (hours)')
plt.ylabel('ln([$MoO_3S$])')
#plt.show()
plt.close()

test1 = time_hours[600:].reshape(-1,1)
test2 = np.log10(C_solutions_mean[600:])
C_log_regression  = LinearRegression().fit(test1, test2)

plt.plot(time_hours[600:], np.log10(C_solutions_mean[600:]), c='chocolate', zorder=1, alpha=1, marker='o')
plt.plot(test1, C_log_regression.predict(test1), c='black', zorder=2, alpha=1)
plt.savefig('ln_C_C.png', bbox_inches='tight', pad_inches=0.075, dpi=200, alpha=0.004)
plt.xlabel('Time (hours)')
plt.ylabel('ln([$MoO_2S_2$])')
#plt.show()
plt.close()

test1 = time_hours[200:].reshape(-1,1)
test2 = np.log10(A_o - D_solutions_mean[200:])
D_log_regression  = LinearRegression().fit(test1, test2)

plt.plot(time_hours[200:], np.log10(A_o - D_solutions_mean[200:]), c='royalblue', zorder=1, alpha=1, marker='o')
plt.plot(test1, D_log_regression.predict(test1), c='black', zorder=2, alpha=1)
plt.savefig('ln_C_D.png', bbox_inches='tight', pad_inches=0.075, dpi=200, alpha=0.004)
plt.xlabel('Time (hours)')
plt.ylabel('ln($\Sigma[Mo] - [$MoOS_3$])')
#plt.show()
plt.close()

test1 = time_hours[200:].reshape(-1,1)
test2 = np.log10(A_o - E_solutions_mean[200:])
E_log_regression  = LinearRegression().fit(test1, test2)

plt.plot(time_hours[200:], np.log10(A_o - E_solutions_mean[200:]), c='dodgerblue', zorder=1, alpha=1, marker='o')
plt.plot(test1, E_log_regression.predict(test1), c='black', zorder=2, alpha=1)
plt.savefig('ln_C_E.png', bbox_inches='tight', pad_inches=0.075, dpi=200, alpha=0.004)
plt.xlabel('Time (hours)')
plt.ylabel('ln([$\Sigma[Mo] - [$MoS_4$])')
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
