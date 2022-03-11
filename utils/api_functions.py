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
        LocalOpt.Cycles = ZOSAPI.Tools.Optimization.OptimizationCycles.Fixed_1_Cycles
    
    LocalOpt.NumberOfCores = cores
    LocalOpt.RunAndWaitForCompletion()
    LocalOpt.Close()

    return


def local_optimisation_n_cycles(TheSystem, ZOSAPI, algorithm = 'DLS', cycles = 50, cores = 8):
     
    # There seems to be a bug right now where optics studio just crashes.

    num_50s = cycles // 50 
    rest = (cycles % 50)/10 #currently should work for multiples of 10

    for i in range(num_50s):
        local_optimisation(TheSystem, ZOSAPI,algorithm,50,cores)
    
    for i in range(rest):
        local_optimisation(TheSystem, ZOSAPI,algorithm,10,cores)
    
    return