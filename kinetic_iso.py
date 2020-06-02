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
from time import sleep
from progress.bar import Bar
import pandas as pd
import os
#from scipy.interpolate import spline

###############################################################################
###############################################################################
###############################################################################

#define a function to calculate and test inputs

def combo_finder(frac, iso, choices):
    iso_calc = np.zeros((5, choices))
    for z, i in enumerate(frac):
        iso_calc[z] = frac[z] * iso[z]
    return iso_calc

###############################################################################
###############################################################################
###############################################################################

#function to test the output from combo_finder
def solution_tester(input, solution):
    tst_output = np.zeros((len(input)))
    for idx, i in enumerate(input):
        test = sum(input[idx])
        tst_output[idx] = test
    return  tst_output

###############################################################################
###############################################################################
###############################################################################

#function to use previous input if near correct value and try again
#pseudo machine learning
def solution_improvement():
    pass

###############################################################################
###############################################################################
###############################################################################

def choose_answers(boundry_array, iterate, offset, choices):

    np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})

    #choose a set of values from the boundries to be tested
    if not iterate:
        A_test = np.linspace(boundry_array[0,0], boundry_array[0,1], choices)
        B_test = np.linspace(boundry_array[1,0], boundry_array[1,1], choices)
        C_test = np.linspace(boundry_array[2,0], boundry_array[2,1], choices)
        D_test = np.linspace(boundry_array[3,0], boundry_array[3,1], choices)
        E_test = np.linspace(boundry_array[4,0], boundry_array[4,1], choices)

    if iterate:
        boundry_low = boundry_array * (1 - offset)
        boundry_high = boundry_array * (1 + offset)

        #make sure that new boundries do not violate original bounds
        for sec in range(2):
            for idx, i in enumerate(Bounds.T[sec]):
                if sec == 0 and i > boundry_low[idx]:
                    print('low  warning pass', idx, i, boundry_low[idx])
                    boundry_low[idx] = Bounds.T[sec, idx]
                if sec == 1 and i < boundry_high[idx]:
                    print('high warning pass', idx, i, boundry_high[idx])
                    boundry_high[idx] = Bounds.T[sec, idx]
        A_test = np.linspace(boundry_low[0], boundry_high[0], choices)
        B_test = np.linspace(boundry_low[1], boundry_high[1], choices)
        C_test = np.linspace(boundry_low[2], boundry_high[2], choices)
        D_test = np.linspace(boundry_low[3], boundry_high[3], choices)
        E_test = np.linspace(boundry_low[4], boundry_high[4], choices)

    #randomly (if you want) choose some values from this setup
    #pick some random numbers from linspace and assign them to an array
    rand_A = np.array([np.random.choice(choices) for i in range(choices)])
    rand_B = np.array([np.random.choice(choices) for i in range(choices)])
    rand_C = np.array([np.random.choice(choices) for i in range(choices)])
    rand_D = np.array([np.random.choice(choices) for i in range(choices)])
    rand_E = np.array([np.random.choice(choices) for i in range(choices)])

    #assign randomized values to arrays
    A_randomized = A_test[rand_A]
    B_randomized = B_test[rand_B]
    C_randomized = C_test[rand_C]
    D_randomized = D_test[rand_D]
    E_randomized = E_test[rand_E]

    return np.array([A_randomized, B_randomized, C_randomized, D_randomized, E_randomized])

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
def Mo_solver(Sig_S, eq, divtime):
    A = A_o * np.exp(-ki*divtime) #+ min

    B = np.divide((ki*A_o),(kii-ki)) * (np.exp(-ki*divtime)-np.exp(-kii*divtime)) #+ min

    C = (np.divide((kii*ki*A_o),(kii-ki)) * \
        ((np.divide(1,(kiii-ki)) * np.exp(-ki*divtime)) - \
        (np.divide(1,(kiii-kii)) * (np.exp(-kii*divtime))) + \
        (np.divide(1 ,(kiii-kii)) * (np.exp(-kiii*divtime))) - \
        (np.divide(1 ,(kiii-ki)) * (np.exp(-kiii*divtime)))))

    D = (np.divide((kiii*kii*ki*A_o), (kii-ki))  * \
        ((np.divide(1, ((kiii-ki)*(kiiii-ki))) * np.exp(-ki*divtime)) - \
        (np.divide(1, ((kiii-kii)*(kiiii-kii))) * np.exp(-kii*divtime)) + \
        (np.divide(1, ((kiii-kii)*(kiiii-kiii))) * np.exp(-kiii*divtime)) - \
        (np.divide(1, ((kiii-ki)*(kiiii-kiii))) * np.exp(-kiii*divtime)) - \
        (np.divide(1, ((kiii-ki)*(kiiii-ki))) * np.exp(-kiiii*divtime)) + \
        (np.divide(1, ((kiii-kii)*(kiiii-kii))) * np.exp(-kiiii*divtime)) - \
        (np.divide(1, ((kiii-kii)*(kiiii-kiii))) * np.exp(-kiiii*divtime)) + \
        (np.divide(1, ((kiii-ki)*(kiiii-kiii))) * np.exp(-kiiii*divtime))))

    E = A_o + (np.divide((kiiii*kiii*kii*ki*A_o), (kii-ki)) * \
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

#calculate the mole fraction of each species
def Mo_mole_frac(input):
    Mo_calc = input/User_Mo
    return Mo_calc

###############################################################################
###############################################################################
###############################################################################

def yes_or_no(question):
    while "the answer is invalid":
        reply = str(input(question+' (y/n): ')).lower().strip()
        if reply[0] == 'y':
            return True
        if reply[0] == 'n':
            return False

###############################################################################
###############################################################################
###############################################################################
external_list = yes_or_no('use profile file?   ')

if external_list:
    df  = pd.read_excel('black_sea_NW.xlsx',header= 0)
#run = int(input('what line to use?   '))

#for xrun in range(2, df.shape[0]):
for xrun in range(2,3):

    if external_list:

        number_of_samples = df.iterations.values[xrun]
        User_Mo = df.Mo.values[xrun]
        User_pH = df.pH.values[xrun]
        User_Sulfide = df.sulfide.values[xrun]
        Save_directory = df.directory.values[xrun]
        profile_graph = df.calculation.values[xrun]
        iso_offset = df.offset.values[xrun]
        User_depth = df.depth.values[xrun]
        Mo_removal = df.removal.values[xrun]
        user_threshold = 0.5
    else:
        number_of_samples = int(input('number of iterations?  '))
        User_Mo = float(input('estimated total Mo conc. (M)?  '))
        User_Sulfide = float(input('estimated sulfide conc. (M)?  '))
        User_pH = float(input('estimated pH value?  '))
        user_threshold = float(input('threashold variable for isotope calculation  '))
        profile_graph = yes_or_no('calculate water column profile?  ')
        Save_directory = 'result/'
        os.makedir('result')
        if profile_graph:
            iso_offset = float(input('mean watercolumn isotope value (e.g., seawater ~ +2.25)?   '))
            Mo_removal = float(input('percent Mo removed for water depth?   ')) / 100

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

#set the boundry conditions of the equations & define some global constants
    Bounds = np.array([[0,5], [0,5], [-5,5], [-5,0], [-5,-1]])
    real_value = 0.05

###############################################################################
###############################################################################
###############################################################################

#k values are based on published experiments in: Clarke et al., (1987) Kinetics of the formation and hydrolysis
#reaction of some thiomolybdate(VI) anions in aqueous solution. Inorg. Chim. Acta
#and Harmer & Sykes (1980). Kinetics of the Interconversion of Sulfido- and
#Oxomolybdate(VI) Species MoOxS4-x2- in Aqueous Solutions. Inorganic Chemistry

    ki_low = 1e-4
    kii_low = 4e-6
    kiii_low = 12.798 * 0.8 * np.exp(-2.116 * User_pH)
    kiiii_low = 0.04305 * np.exp(-1.431 * User_pH)

    ki_hi = 3e-4
    kii_hi = 6e-6
    kiii_hi = 12.798 * 1.2 * np.exp(-2.116 * User_pH)
    kiiii_hi = 0.04305 * 1.5 * np.exp(-1.431 * User_pH)

    ki_range = np.linspace(ki_low,ki_hi,10)
    kii_range = np.linspace(kii_low,kii_hi,10)
    kiii_range = np.linspace(kiii_low,kiii_hi,10)
    kiiii_range = np.linspace(kiiii_low,kiiii_hi,10)

#sulfide_Mo_ratio = User_Sulfide / User_Mo
#for i, z in enumerate(kii_range):
#   ki_range[i] =  ki_range[i] * (0.427**sulfide_Mo_ratio / ki_range[i])
#   kii_range[i] = kii_range[i] * (0.2999**sulfide_Mo_ratio / kii_range[i])


###############################################################################
###############################################################################
###############################################################################

#time_values = range_inc(0,1000,1,10)
    time_values = np.linspace(0, 500000, 1000)[0:]

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

#add noise range to the sulfide values
    Sulfide_low = User_Sulfide / 1.1
    Sulfide_high = User_Sulfide * 1.1

#set the distribution of possible sulfide values
    Rand_Sulfide = np.array([np.random.uniform(Sulfide_low, Sulfide_high) for i in range(number_of_samples)])

#Create loop to calculate each equation, using k values define above
    for indexer, (ki, kii, kiii, kiiii) in enumerate(zip(ki_randomized, kii_randomized, kiii_randomized, kiiii_randomized)):

        for sec_indexer, (t, A_o) in enumerate(zip(time_values, Ao_realvalues)):

            A_solutions[indexer, sec_indexer] = Mo_solver(Rand_Sulfide[indexer], 1, time_values[sec_indexer])
            B_solutions[indexer, sec_indexer] = Mo_solver(Rand_Sulfide[indexer], 2, time_values[sec_indexer])
            C_solutions[indexer, sec_indexer] = Mo_solver(Rand_Sulfide[indexer], 3, time_values[sec_indexer])
            D_solutions[indexer, sec_indexer] = Mo_solver(Rand_Sulfide[indexer], 4, time_values[sec_indexer])
            E_solutions[indexer, sec_indexer] = Mo_solver(Rand_Sulfide[indexer], 5, time_values[sec_indexer])

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
#TO DO LIST
#make this section cleaner with a function called for each species - SRH
#create loop with inputs needed and run overnight
#save r^2 vaules for each run sig can be calculated by 1000 r values assign confidence intervals
# two tailed test (multiply by 0.025 and 0.975 for upper and lower bounds if R is in a tail gives significance)
#can create model distribution of all runs
#create a choice to calculate isotope species from known total, or calculate the isotope profile from concentration data with species
#add tolerance variable for isotope values that are not close to zero or create logic to use for each case?

    tester_array = np.zeros((number_of_samples, len(time_values)))
    data_holder = np.zeros((5, len(time_values), number_of_samples))
    test0 = np.zeros((1000))
    test1 = np.zeros((1000))
    test2 = np.zeros((1000))
    test3 = np.zeros((1000))
    test4 = np.zeros((1000))

    for run in range(number_of_samples):
        a_temp = []
        b_temp = []
        c_temp = []
        d_temp = []
        e_temp = []
        bar = Bar('Processing', max=1000)

        for idx in range(1000):
            a_temp.clear()
            b_temp.clear()
            c_temp.clear()
            d_temp.clear()
            e_temp.clear()
            sleep(0.1)
            bar.next()
            mole_ratio = [A_mole_ratio[run,idx], B_mole_ratio[run,idx], C_mole_ratio[run,idx], D_mole_ratio[run,idx], E_mole_ratio[run,idx]]
            tester_array = choose_answers(Bounds, False, 0.4, 100)
            final_values = combo_finder(mole_ratio, tester_array, 100)
            final_values = final_values.T
            results = solution_tester(final_values, real_value)
        #data_holder[idx] = results
            new_boundry = np.zeros((5))
            threshold_value = real_value * user_threshold
    #store the index for each result that is close to real value, to later use a solution
            for z, i in enumerate(results):
                if i < (real_value * 2) and i > (real_value * -2):
                    a_temp.append(tester_array.T[z, 0])
                    b_temp.append(tester_array.T[z, 1])
                    c_temp.append(tester_array.T[z, 2])
                    d_temp.append(tester_array.T[z, 3])
                    e_temp.append(tester_array.T[z, 4])

            if len(a_temp) == 0:
                a_temp.append(tester_array.T[:, 0])
                b_temp.append(tester_array.T[:, 1])
                c_temp.append(tester_array.T[:, 2])
                d_temp.append(tester_array.T[:, 3])
                e_temp.append(tester_array.T[:, 4])
            else:
                data_holder[0, idx, run] = np.mean(a_temp)
                data_holder[1, idx, run] = np.mean(b_temp)
                data_holder[2, idx, run] = np.mean(c_temp)
                data_holder[3, idx, run] = np.mean(d_temp)
                data_holder[4, idx, run] = np.mean(e_temp)

    for i, z in enumerate(data_holder[:,0,0]):
        new_boundry[i] = data_holder[i,:,0].mean()

    for z, i in enumerate(data_holder[0,:,0]):
        test0[z] = data_holder[0,z,:].mean()
        test1[z] = data_holder[1,z,:].mean()
        test2[z] = data_holder[2,z,:].mean()
        test3[z] = data_holder[3,z,:].mean()
        test4[z] = data_holder[4,z,:].mean()

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


    plt.scatter(A_model_corr[:], A_emperical[:], c='grey', marker='o')
    plt.scatter(B_model_corr[:], B_emperical[:], c='goldenrod', marker='o')
    plt.scatter(C_model_corr[:], C_emperical[:], c='chocolate', marker='o')
    plt.scatter(D_model_corr[:], D_emperical[:], c='royalblue', marker='o')
    plt.scatter(E_model_corr[:], E_emperical[:], c='dodgerblue', marker='o')
    plt.ylim(0, 0.0001)
    plt.xlim(0, 0.0001)
    plt.savefig((Save_directory + 'correlation.png'), bbox_inches='tight', pad_inches=0.075, dpi=200)
    #plt.show()
    plt.close()

    conc_regress = []
#Test regression between model data and experimental data
    A_regression  = LinearRegression().fit(A_prime, A_emperical)
#print('\nA model verus real conc. r^2', A_regression.score(A_prime, A_emperical))
    conc_regress.append((A_regression.score(A_prime, A_emperical)))

    B_regression  = LinearRegression().fit(B_prime, B_emperical)
#print('B model verus real conc. r^2', B_regression.score(B_prime, B_emperical))
    conc_regress.append((B_regression.score(B_prime, B_emperical)))

    C_regression  = LinearRegression().fit(C_prime, C_emperical)
#print('C model verus real conc. r^2', C_regression.score(C_prime, C_emperical))
    conc_regress.append((C_regression.score(C_prime, C_emperical)))

    D_regression  = LinearRegression().fit(D_prime, D_emperical)
#print('D model verus real conc. r^2', D_regression.score(D_prime, D_emperical))
    conc_regress.append((D_regression.score(D_prime, D_emperical)))

    E_regression  = LinearRegression().fit(E_prime, E_emperical)
#print('E model verus real conc. r^2', E_regression.score(E_prime, E_emperical))
    conc_regress.append((E_regression.score(E_prime, E_emperical)))

###############################################################################
###############################################################################
###############################################################################

#verification of isotopic ratios
    A_model_corr = []
    B_model_corr = []
    C_model_corr = []
    D_model_corr = []
    E_model_corr = []

#create specific lists for testing correlation for each species
    for i, q in time_array:
        A_model_corr.append(data_holder[0,i,0])
        B_model_corr.append(data_holder[1,i,0])
        C_model_corr.append(data_holder[2,i,0])
        D_model_corr.append(data_holder[3,i,0])
        E_model_corr.append(data_holder[4,i,0])

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

#make a scatter plot of model output versus experimental data
    plt.scatter(A_model_corr[:], A_iso_emperical[:], c='grey', marker='o')
    plt.scatter(B_model_corr[1:], B_iso_emperical[1:], c='goldenrod', marker='o')
    plt.scatter(C_model_corr[:], C_iso_emperical[:], c='chocolate', marker='o')
    plt.scatter(D_model_corr[:], D_iso_emperical[:], c='royalblue', marker='o')
    plt.scatter(E_model_corr[:], E_iso_emperical[:], c='dodgerblue', marker='o')
    plt.ylim(0)
    plt.xlim(0, np.max(A_model_corr))
    plt.savefig((Save_directory +'correlation_iso.png'), bbox_inches='tight', pad_inches=0.075, dpi=200)
    #plt.show()
    plt.close()

    iso_regress = []
#test regression of model data versus experimental data
    print('\n')
    A_regression  = LinearRegression().fit(A_prime[2:], A_iso_emperical[2:])
    #print('A model verus real isotopes. r^2', A_regression.score(A_prime[2:], A_iso_emperical[2:]))
    iso_regress.append((A_regression.score(A_prime[2:], A_iso_emperical[2:])))

    B_regression  = LinearRegression().fit(B_prime[2:], B_iso_emperical[2:])
    #print('B model verus real isotopes r^2', B_regression.score(B_prime[2:], B_iso_emperical[2:]))
    iso_regress.append((B_regression.score(B_prime[2:], B_iso_emperical[2:])))

    C_regression  = LinearRegression().fit(C_prime[2:], C_iso_emperical[2:])
    #print('C model verus real isotopes. r^2', C_regression.score(C_prime[2:], C_iso_emperical[2:]))
    iso_regress.append((C_regression.score(C_prime[2:], C_iso_emperical[2:])))

    D_regression  = LinearRegression().fit(D_prime[2:], D_iso_emperical[2:])
    #print('D model verus real isotopes r^2', D_regression.score(D_prime[2:], D_iso_emperical[2:]))
    iso_regress.append((D_regression.score(D_prime[2:], D_iso_emperical[2:])))

    E_regression  = LinearRegression().fit(E_prime[2:], E_iso_emperical[2:])
    #print('E model verus real isotopes. r^2', E_regression.score(E_prime[2:], E_iso_emperical[2:]))
    iso_regress.append((E_regression.score(E_prime[2:], E_iso_emperical[2:])))

###############################################################################
###############################################################################
###############################################################################

    #save the regression data into csv file
    np.savetxt((Save_directory +'regressions.csv'),np.column_stack((conc_regress, iso_regress)), delimiter=',', fmt='%.4f', header='concentration, isotopes')

###############################################################################
###############################################################################
###############################################################################

#make figure layout
    fig = plt.figure(figsize=(8.,4.))

#set axes for the plot within the figure
    ax = fig.add_axes([0.0, 0., 1., 1.])

    for i in range(0,number_of_samples):

        ls2, = ax.plot(time_values[:], A_solutions[i,:], c='grey', zorder=5, alpha=0.3)
        ls3, = ax.plot(time_values[:], B_solutions[i,:], c='goldenrod', zorder=4, alpha=0.3)
        ls4, = ax.plot(time_values[:], C_solutions[i,:], c='chocolate', zorder=3, alpha=0.3)
        ls5, = ax.plot(time_values[:], D_solutions[i,:], c='royalblue', zorder=2, alpha=0.3)
        ls6, = ax.plot(time_values[:], E_solutions[i,:], c='dodgerblue', zorder=1, alpha=0.3)

    plt.margins(x=0, y=0)

    ax.set_xlabel(r'Time (s; $ \times 10^{5}$)', fontsize=12, color='k')
    ax.set_ylabel(r'[Mo]', fontsize=12, color='k')

    line1 = pylab.Line2D(range(10),range(10),marker="_",linewidth=2.0,color="grey",alpha=1.0)
    line2 = pylab.Line2D(range(10),range(10),marker="_",linewidth=2.0,color="goldenrod",alpha=1.0)
    line3 = pylab.Line2D(range(10),range(10),marker="_",linewidth=2.0,color="chocolate",alpha=1.0)
    line4 = pylab.Line2D(range(10),range(10),marker="_",linewidth=2.0,color="royalblue",alpha=1.0)
    line5 = pylab.Line2D(range(10),range(10),marker="_",linewidth=2.0,color="dodgerblue",alpha=1.0)

    ax.legend([line1, line2, line3, line4, line5], \
        ['MoO$_{4}$','MoO$_{3}$S','MoO$_{2}$S$_{2}$','MoOS$_{3}$', 'MoS$_{4}$'], \
        loc='upper right', fancybox=True, ncol=2, fontsize=12, framealpha=0.9)

    ax.set_xticks([0.,100000,200000,300000,400000,500000])
    ax.set_xticklabels(['0','1','2','3','4','5'])

    ax.grid(which='major', axis='both', linestyle='--', alpha=0.5)
    plt.savefig((Save_directory +'Mo_species_conc.png'), bbox_inches='tight', pad_inches=0.075, dpi=200, alpha=0.004)
    #plt.show()
    plt.close()

###############################################################################
###############################################################################
###############################################################################

    time_hours = time_values / 3600

    lin_test1 = time_hours[50:].reshape(-1,1)
    lin_test2 = np.log10(B_solutions_mean[50:])
    B_log_regression  = LinearRegression().fit(lin_test1, lin_test2)

    plt.plot(time_hours[50:], np.log10(B_solutions_mean[50:]), c='goldenrod', zorder=1, alpha=1, marker='o')
    plt.plot(lin_test1, B_log_regression.predict(lin_test1), c='black', zorder=2, alpha=1)
    plt.xlim(0)
    plt.xlabel('Time (hours)')
    plt.ylabel('ln([$MoO_3S$])')
    plt.savefig((Save_directory +'ln_C_B.png'), bbox_inches='tight', pad_inches=0.075, dpi=200, alpha=0.004)
    #plt.show()
    plt.close()

    lin_test1 = time_hours[600:].reshape(-1,1)
    lin_test2 = np.log10(C_solutions_mean[600:])
    C_log_regression  = LinearRegression().fit(lin_test1, lin_test2)

    plt.plot(time_hours[600:], np.log10(C_solutions_mean[600:]), c='chocolate', zorder=1, alpha=1, marker='o')
    plt.plot(lin_test1, C_log_regression.predict(lin_test1), c='black', zorder=2, alpha=1)
    plt.xlabel('Time (hours)')
    plt.ylabel('ln([$MoO_2S_2$])')
    plt.savefig((Save_directory +'ln_C_C.png'), bbox_inches='tight', pad_inches=0.075, dpi=200, alpha=0.004)
    #plt.show()
    plt.close()

    lin_test1 = time_hours[200:].reshape(-1,1)
    lin_test2 = np.log10(A_o - D_solutions_mean[200:])
    D_log_regression  = LinearRegression().fit(lin_test1, lin_test2)

    plt.plot(time_hours[200:], np.log10(A_o - D_solutions_mean[200:]), c='royalblue', zorder=1, alpha=1, marker='o')
    plt.plot(lin_test1, D_log_regression.predict(lin_test1), c='black', zorder=2, alpha=1)
    plt.xlabel('Time (hours)')
    plt.ylabel('ln($\Sigma[Mo] - [$MoOS_3$])')
    plt.savefig((Save_directory +'ln_C_D.png'), bbox_inches='tight', pad_inches=0.075, dpi=200, alpha=0.004)
    #plt.show()
    plt.close()

    lin_test1 = time_hours[200:].reshape(-1,1)
    lin_test2 = np.log10(A_o - E_solutions_mean[200:])
    E_log_regression  = LinearRegression().fit(lin_test1, lin_test2)

    plt.plot(time_hours[200:], np.log10(A_o - E_solutions_mean[200:]), c='dodgerblue', zorder=1, alpha=1, marker='o')
    plt.plot(lin_test1, E_log_regression.predict(lin_test1), c='black', zorder=2, alpha=1)
    plt.xlabel('Time (hours)')
    plt.ylabel('ln([$\Sigma[Mo] - [$MoS_4$])')
    plt.savefig((Save_directory +'ln_C_E.png'), bbox_inches='tight', pad_inches=0.075, dpi=200, alpha=0.004)
    #plt.show()
    plt.close()

###############################################################################
###############################################################################
###############################################################################

    #make figure layout
    fig = plt.figure(figsize=(8.,4.))

    #set axes for the plot within the figure
    ax = fig.add_axes([0.0, 0., 1., 1.])

    for i in range(0,number_of_samples):
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

    ax.legend([line1, line2, line3, line4, line5], \
        ['MoO$_{4}$','MoO$_{3}$S','MoO$_{2}$S$_{2}$','MoOS$_{3}$', 'MoS$_{4}$'], \
        loc='upper right', fancybox=True, ncol=2, fontsize=12, framealpha=0.9)

    ax.set_xticks([0.,100000,200000,300000,400000,500000])
    ax.set_xticklabels(['0','1','2','3','4','5'])

    ax.set_yticks([0.0,0.1,0.3,0.5,0.8,1])
    ax.set_yticklabels(['0','10','30','50','80','100'], fontsize=12)

    ax.grid(which='major', axis='both', linestyle='--', alpha=0.5)

    plt.savefig((Save_directory +'Mo_species_percent.png'), bbox_inches='tight', pad_inches=0.075, dpi=200, alpha=0.004)
    #plt.show()
    plt.close()

###############################################################################
###############################################################################
###############################################################################

#define a horizontal line at 0 on y axis
    fig = plt.figure(figsize=(8.,4.))

    #set axes for the plot within the figure
    ax = fig.add_axes([0.0, 0., 1., 1.])

    for i in range(0,number_of_samples):
        ls2, = ax.plot(time_values[:], data_holder[0,:,i], c='grey', zorder=5, alpha=0.3)
        ls3, = ax.plot(time_values[:], data_holder[1,:,i], c='goldenrod', zorder=4, alpha=0.3)
        ls4, = ax.plot(time_values[:], data_holder[2,:,i], c='chocolate', zorder=3, alpha=0.3)
        ls5, = ax.plot(time_values[:], data_holder[3,:,i], c='royalblue', zorder=2, alpha=0.3)
        ls6, = ax.plot(time_values[:], data_holder[4,:,i], c='dodgerblue', zorder=1, alpha=0.3)

    ln_horizontal = np.zeros(len(time_values))

    plt.plot(time_values, ln_horizontal, marker='', color='black', linestyle='--', markeredgecolor='black', linewidth=0.5)

#plt.xlim(0,500000)

    ax.set_xticks([0.,100000,200000,300000,400000,500000])
    ax.set_xticklabels(['0','1','2','3','4','5'])
    ax.set_xlabel(r'Time (s; $ \times 10^{5}$)', fontsize=12, color='k')
    ax.set_ylabel(r'[$^{98}$Mo] (permille)', fontsize=12, color='k')

    ax.legend([line1, line2, line3, line4, line5], \
        ['MoO$_{4}$','MoO$_{3}$S','MoO$_{2}$S$_{2}$','MoOS$_{3}$', 'MoS$_{4}$'], \
        loc='upper right', fancybox=True, ncol=2, fontsize=12, framealpha=0.9)

    plt.ylim(-5,5)

    plt.savefig((Save_directory +'image_delta.png'), bbox_inches='tight', pad_inches=0.075, dpi=200, alpha=0.004)
    #plt.show()
    plt.close()

###############################################################################
###############################################################################
###############################################################################

    #define a horizontal line at 0 on y axis
    fig = plt.figure(figsize=(8.,4.))

    #set axes for the plot within the figure
    ax = fig.add_axes([0.0, 0., 1., 1.])

    for i in range(0,number_of_samples):
        ls2, = ax.plot(time_values[:], test0[:], c='grey', zorder=5, alpha=0.3)
        ls3, = ax.plot(time_values[:], test1[:], c='goldenrod', zorder=4, alpha=0.3)
        ls4, = ax.plot(time_values[:], test2[:], c='chocolate', zorder=3, alpha=0.3)
        ls5, = ax.plot(time_values[:], test3[:], c='royalblue', zorder=2, alpha=0.3)
        ls6, = ax.plot(time_values[:], test4[:], c='dodgerblue', zorder=1, alpha=0.3)

    ln_horizontal = np.zeros(len(time_values))

    plt.plot(time_values, ln_horizontal, marker='', color='black', linestyle='--', markeredgecolor='black', linewidth=0.5)

    #plt.xlim(0,500000)

    ax.set_xticks([0.,100000,200000,300000,400000,500000])
    ax.set_xticklabels(['0','1','2','3','4','5'])

    ax.set_xlabel(r'Time (s; $ \times 10^{5}$)', fontsize=12, color='k')
    ax.set_ylabel(r'[$^{98}$Mo] (permille)', fontsize=12, color='k')

    ax.legend([line1, line2, line3, line4, line5], \
        ['MoO$_{4}$','MoO$_{3}$S','MoO$_{2}$S$_{2}$','MoOS$_{3}$', 'MoS$_{4}$'], \
        loc='upper right',  fancybox=True, ncol=2, fontsize=12, framealpha=0.9)

    plt.ylim(-5,5)

    plt.savefig((Save_directory +'image_delta_mean.png'), bbox_inches='tight', pad_inches=0.075, dpi=200, alpha=0.004)
    #plt.show()
    plt.close()

###############################################################################
###############################################################################
###############################################################################

    #save all the data for the mean isotope results
    np.savetxt((Save_directory +'mean_isotope.csv'),np.column_stack((test0, test1, test2, test3, test4)), \
        delimiter=',', fmt='%.3f', header="A, B, C, D, E")

        #save all the data for the mean isotope results
    np.savetxt((Save_directory +'mean_concentration.csv'),np.column_stack((A_solutions_mean, B_solutions_mean, \
        C_solutions_mean, D_solutions_mean, E_solutions_mean)), \
        header="A, B, C, D, E", delimiter=',')

###############################################################################
###############################################################################
###############################################################################
#model calculation of profile

    pub_test = np.zeros((5))
    test_ratio = np.zeros((5))
    new_isotopic_value = np.zeros((5))

    if profile_graph:
        Mo_removal = Mo_removal / 100

        for i, z in enumerate(mole_ratio):
            test_ratio[i] = mole_ratio[i]

        pub_test[0] = test_ratio[0]*test0[999] + test_ratio[1]*test1[999] + \
            test_ratio[2]*test2[999] + test_ratio[3]*test3[999] + test_ratio[4]*test4[999]

        if test_ratio[4] - Mo_removal > 0:

            pub_test[1] = test_ratio[0]*test0[999] + test_ratio[1]*test1[999] + \
                    test_ratio[2]*test2[999] + test_ratio[3]*test3[999] + (test_ratio[4] - Mo_removal)*test4[999]

            print('depth ', df.depth.values[xrun],'total 98Mo isotopic signature', pub_test[1] + iso_offset)
            iso_output = pub_test[1] + iso_offset

        if test_ratio[4] - Mo_removal < 0:
            if test_ratio[3] + (test_ratio[4] - Mo_removal) < 0:
                new_total = sum(test_ratio)

                for i, k in enumerate(test_ratio):
                    test_ratio[i] = test_ratio[i]/new_total

                pub_test[3] = test_ratio[0]*test0[999] + test_ratio[1]*test1[999] + \
                    (test_ratio[2] - (test_ratio[3] + (test_ratio[4] - Mo_removal)))*test2[999]

                print('depth ', df.depth.values[xrun],'total 98Mo isotopic signature', pub_test[3] + iso_offset)
                iso_output = pub_test[3] + iso_offset

            else:
                new_total = sum(test_ratio)
                for i, k in enumerate(test_ratio):
                    test_ratio[i] = test_ratio[i]/new_total

                pub_test[2] = test_ratio[0]*test0[999] + test_ratio[1]*test1[999] + \
                    test_ratio[2]*test2[999] + (test_ratio[3] + (test_ratio[4] - Mo_removal))*test3[999]

                print('depth ', df.depth.values[xrun], 'total 98Mo isotopic signature', pub_test[2] + iso_offset)
                iso_output = pub_test[2] + iso_offset
#save the regression data into csv file
    np.savetxt((Save_directory + 'iso_fractions.csv'),np.column_stack((conc_regress, iso_regress)), delimiter=',', fmt='%.4f', header='concentration, isotopes')

#repeat the loops to calculate sulfide consumed, truncate data accordingly
Species_cutoff = np.zeros((5, number_of_samples))
dominant_species = np.zeros(shape=(5))

#Check if the species is decreasing, don't double count moles of sulfide consumed
for Sindexer, Svalue in enumerate(Rand_Sulfide):
    #find the index of the max value for each species
    B_idxmax = B_solutions[Sindexer].argmax()
    C_idxmax = C_solutions[Sindexer].argmax()
    D_idxmax = D_solutions[Sindexer].argmax()
    E_idxmax = E_solutions[Sindexer].argmax()

    i=0
    checker = True

    while i < B_idxmax and checker:
        if (Rand_Sulfide[Sindexer] - (B_solutions[Sindexer][i] + C_solutions[Sindexer][i] + D_solutions[Sindexer][i] + E_solutions[Sindexer][i])) <= 0:
            dominant_species[1] +=  1
            Species_cutoff[1, Sindexer] = i
            checker = False
        i += 1

    i = B_idxmax

    while i < C_idxmax and checker:
        if (Rand_Sulfide[Sindexer] - (B_solutions[Sindexer][B_idxmax] + C_solutions[Sindexer][i] + D_solutions[Sindexer][i] + E_solutions[Sindexer][i])) <= 0:
            dominant_species[2] += 1
            Species_cutoff[2, Sindexer] = i
            checker = False
        i+=1

    i = C_idxmax

    while i < D_idxmax and checker:
        if (Rand_Sulfide[Sindexer] - (B_solutions[Sindexer][B_idxmax] + C_solutions[Sindexer][C_idxmax] + D_solutions[Sindexer][i] + E_solutions[Sindexer][i])) <= 0:
            dominant_species[3] += 1
            Species_cutoff[3, Sindexer] = i
            checker = False
        i+=1

    if checker:
        dominant_species[4] += 1
        Species_cutoff[4, Sindexer] = E_idxmax

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
dom_spec = ['MoO$_{4}$','MoO$_{3}$S','MoO$_{2}$S$_{2}$','MoOS$_{3}$', 'MoS$_{4}$']

sum_spec = np.zeros((5))

ax.bar(dom_spec,dominant_species)

plt.savefig((Save_directory +'image_hist.png'), bbox_inches='tight', pad_inches=0.075, dpi=200, alpha=0.004)
#plt.show()
plt.close()
