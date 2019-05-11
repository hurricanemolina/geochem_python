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


##############################################################################  
##############################################################################  
##############################################################################  


#arrange the data into an xarray dataset with data information
data = xr.Dataset({'MoO4':(['time'], file1['MoO4 (M)'].values),
                   'MoO3S':(['time'], file1['MoO3S (M)'].values),
                   'MoO2S2':(['time'], file1['MoO2S2 (M)'].values),
                   'MoOS3':(['time'], file1['MoOS3 (M)'].values),
                   'MoS4':(['time'], file1['MoS4 (M)'].values)},
                   coords={'time':file1['time (s)'].values},
                   attrs={'File Contents':'Concentration Data',
                          'Data Owner':'Stephan Hlohowskyj',
                          'Time Units':'Seconds (s)'})

#save the dataset into a netCDF file for later use in Step 1
data.to_netcdf('file1_new_stephan.nc')


##############################################################################  
##############################################################################  
##############################################################################  


#repeat steps above for file2 to create new netCDF file.

#file2 = pd.read_excel('file2_stephan.xlsx', skiprows=1)


# ...


##############################################################################  
##############################################################################  
##############################################################################  

