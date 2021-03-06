'''
Convenience functions for API...
'''


import numpy as np
import matplotlib.pyplot as plt

import os 
import sys
import h5py



def double2array(data):
    '''
    Needed to extract the values out of radius/thickness arrays
    '''
    return np.array(list(data))

def get_parameter(LDE):
    '''Returns thicknesses and Radii for all surfaces in separated arrays'''

    thicknesses = []
    radii = []
    for i in np.arange(LDE.NumberOfSurfaces):

        obj = LDE.GetSurfaceAt(i)
        thicknesses.append(obj.Thickness)
        radii.append(obj.Radius)
    #return thicknesses, radii
    return np.array(thicknesses), np.array(radii)



def set_parameter(LDE,set_array):
    '''
    Here you will give an array of all the values to be changed. 
    If the value is None, do not change it. 
    
    Column 0 = Element number (starts at index 0)
    Column 1 = Radius in object units
    Column 2 = Thickness in object units
    '''

    if len(set_array.shape) == 1: #in this case they are just changing one param.
        set_array = set_array.reshape((1,3))

    if set_array.shape[1] != 3:
        raise ValueError('Format must be: Column 0 = Element number, Column 1 = Radius, Column 2 = Thickness')
    

    for i in range(len(set_array)):
        obj = LDE.GetSurfaceAt(set_array[i,0])  #this step takes no time.

        if set_array[i,1] is not None:
            obj.RadiusCell.DoubleValue = set_array[i,1]

        if set_array[i,2] is not None:
            obj.ThicknessCell.DoubleValue = set_array[i,2]
    
    return


def make_variable(LDE,var_array):
    '''
    Here you will give an array of all the parameters to be changed. 
    if it is 0, make it fixed;
    if it is 1, make it variable.
    
    Column 0 = Element number (starts at index 0)
    Column 1 = Change Radius? 
    Column 2 = Change Thickness?
    '''
    
    if len(var_array.shape) == 1: #in this case they are just changing one param.
        var_array = var_array.reshape((1,3))

    if var_array.shape[1] != 3:
        raise ValueError('Format must be: Column 0 = Element number, Column 1 = Radius, Column 2 = Thickness')
    



    for i in range(len(var_array)):
        obj = LDE.GetSurfaceAt(var_array[i,0])  #this step takes no time.

        if var_array[i,1] == 1:
            obj.RadiusCell.MakeSolveVariable()
        
        elif var_array[i,1] == 0:
            obj.RadiusCell.MakeSolveFixed()


        if var_array[i,2] == 1:
            obj.ThicknessCell.MakeSolveVariable()
        
        elif var_array[i,2] == 0:
            obj.RadiusCell.MakeSolveFixed()
    
    return


def calc_merit(MFE):
    '''
    Calculate the merit function. 
    '''
    return MFE.CalculateMeritFunction()


def load_merit(MFE, filename):
    '''
    load a merit function. 
    '''
    return MFE.LoadMeritFunction(filename)


def fast_system(TheSystem):
    '''
    By changing update mode we stop zemax updating all graphs
    etc, and it speeds up the parameter changing
    '''
    TheSystem.UpdateMode = 0 
    TheSystem.UpdateStatus()
    return



def local_optimisation(TheSystem, ZOSAPI, algorithm = 'DLS', cycles = 'automatic', cores = 8):
    '''
    run a local optimisation.
    '''
    
    poss_values = ['automatic',1,5,10,50]

    if cycles not in poss_values:
        raise ValueError('If not a standard value, instead use local_optimization_n_cycles') 
    
    LocalOpt = TheSystem.Tools.OpenLocalOptimization()
    if algorithm == 'DLS':
        LocalOpt.Algorithm = ZOSAPI.Tools.Optimization.OptimizationAlgorithm.DampedLeastSquares
    elif algorithm == 'OD':
        LocalOpt.Algorithm = ZOSAPI.Tools.Optimization.OptimizationAlgorithm.OrthogonalDescent
    

    if cycles == 'automatic':
        LocalOpt.Cycles = ZOSAPI.Tools.Optimization.OptimizationCycles.Automatic
    elif cycles == 50:
        LocalOpt.Cycles = ZOSAPI.Tools.Optimization.OptimizationCycles.Fixed_50_Cycles
    elif cycles == 10:
        LocalOpt.Cycles = ZOSAPI.Tools.Optimization.OptimizationCycles.Fixed_10_Cycles
    elif cycles == 5:
        LocalOpt.Cycles = ZOSAPI.Tools.Optimization.OptimizationCycles.Fixed_5_Cycles
    elif cycles == 1:
        LocalOpt.Cycles = ZOSAPI.Tools.Optimization.OptimizationCycles.Fixed_1_Cycle
    
    LocalOpt.NumberOfCores = cores
    LocalOpt.RunAndWaitForCompletion()
    LocalOpt.Close()

    return

def local_optimisation_n_single_cycles(TheSystem, ZOSAPI, algorithm = 'DLS', cycles = 10, cores = 8):
    """Allows to extract the samplings points during optimization"""

    merit_values = np.zeros(cycles)
    
    thickness_values = np.zeros((cycles, TheSystem.LDE.NumberOfSurfaces))
    radius_values = np.zeros((cycles, TheSystem.LDE.NumberOfSurfaces))

    for i in np.arange(cycles):
        local_optimisation(TheSystem, ZOSAPI,algorithm,1,cores)
        thickness_values[i,:], radius_values[i,:] = get_parameter(TheSystem.LDE)
        merit_values[i] = calc_merit(TheSystem.MFE)

    return thickness_values, radius_values, merit_values

def local_optimisation_n_cycles(TheSystem, ZOSAPI, algorithm = 'DLS', cycles = 50, cores = 8):

    # There seems to be a bug right now where optics studio just crashes.

    num_50s = cycles // 50 
    rest = (cycles % 50)/10 #currently should work for multiples of 10

    for i in range(int(num_50s)):
        local_optimisation(TheSystem, ZOSAPI,algorithm,50,cores)
    
    for i in range(int(rest)):
        local_optimisation(TheSystem, ZOSAPI,algorithm,10,cores)
    
    return

def get_variable_status(LDE):
    """Gets two arrays with True/ False flags whether a thickness/radius parameter is a variable in ZOS or not."""
    thickness_var = []
    radius_var = []

    for i in np.arange(LDE.NumberOfSurfaces-1):
        variable_lens = LDE.GetSurfaceAt(i)

        if variable_lens.RadiusCell.Solve == 2:
            radius_var.append(True)
        else:
            radius_var.append(False)
        
        if variable_lens.ThicknessCell.Solve == 2:
            thickness_var.append(True)
        else:
            thickness_var.append(False)  
    return radius_var, thickness_var 
