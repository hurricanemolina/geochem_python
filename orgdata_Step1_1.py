#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 19:49:09 2019

Authors:

Stephan Hlohowskyj
and
Dr. Maria J. Molina

"""

##############################################################################
##############################################################################
##############################################################################


import pandas as pd
import xarray as xr
import numpy as np


##############################################################################
##############################################################################
##############################################################################


#read the excel file
#skipping the first row since we don't need it here
file1 = pd.read_excel('file1_stephan.xlsx', skiprows=1)

#replace the ND string (no data) with np.nan (numpy's missing data way to allow
#for vectorized operations [makes handling missing data easier for later computations])
file1 = file1.replace('ND', np.nan)

#arrange the data into an xarray dataset with data information
data = xr.Dataset({'A':(['time'], file1['MoO4 (M)'].values),
                   'B':(['time'], file1['MoO3S (M)'].values),
                   'C':(['time'], file1['MoO2S2 (M)'].values),
                   'D':(['time'], file1['MoOS3 (M)'].values),
                   'E':(['time'], file1['MoS4 (M)'].values),
                   'A_o':(['time'], file1['Mo_tot'].values)},
                   coords={'time':file1['time (s)'].values},
                   attrs={'File Contents':'Concentration Data',
                          'Data Owner':'Stephan Hlohowskyj',
                          'Time Units':'Seconds (s)',
                          'A':'MoO4',
                          'B':'MoO3S',
                          'C':'MoO2S2',
                          'D':'MoOS3',
                          'E':'MoS4',
                          'A_o':'Mo_tot'})

#save the dataset into a netCDF file for later use in Step 1
data.to_netcdf('file1_new_stephan.nc')


##############################################################################
##############################################################################
##############################################################################


#repeat steps above for file2 to create new netCDF file.
#Will leave this section alone until we deal with the above data and get it working - SRH

file2 = pd.read_excel('file2_stephan.xlsx', skiprows=1)

file2 = file2.replace('ND', np.nan)

data2 = xr.Dataset({'A_o':(['time'], file2['MoO4 (M)'].values),
                    'A':(['time'], file2['MoO3S (M)'].values),
                    'B':(['time'], file2['MoO2S2 (M)'].values),
                    'C':(['time'], file2['MoOS3 (M)'].values),
                    'D':(['time'], file2['MoS4 (M)'].values)},
                    coords={'time':file2['time (s)'].values},
                    attrs={'File Contents':'Isotope Data',
                           'Data Owner':'Stephan Hlohowskyj',
                           'Time Units':'Seconds (s)',
                           'A_o':'MoO4',
                           'A':'MoO3S',
                           'B':'MoO2S2',
                           'C':'MoOS3',
                           'D':'MoS4'})

data2.to_netcdf('file2_new_stephan.nc')


##############################################################################
##############################################################################
##############################################################################
