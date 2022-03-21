'''
Optimisation functions...
'''

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import dual_annealing
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import ElementwiseProblem
from pymoo.factory import get_reference_directions
from pymoo.optimize import minimize

import os 
import sys
import h5py

path = os.path.dirname(os.path.dirname(os.path.realpath("__file__")))
sys.path.insert(0,path)
from utils.api_functions import *



def calculate_grad(Y,X):
    '''
    To be used with very simple gradient descent algorithm. 
    '''
    return (Y[1] - Y[0])/(X[1] - X[0])

def scale_thickness(thickness, thickness_max):
    """Scales Thicknesses smaller than thickness_max to a range of 0 to 1"""
    return thickness/thickness_max

def rescale_thickness(thickness_sc, thickness_max):
    """Returns to original thicknesses"""
    return thickness_sc*thickness_max

def scale_radius_inverse(radius, scaling):
    """Scales Radii larger than +scaling and smaller than -scaling to a range of 0 to 1. Infinity is scaled to 0.5"""
    return 1./(2*radius*scaling)+0.5

def rescale_radius_inverse(radius_sc, scaling):
    """Returns to original radii"""
    return 1./(2*scaling*(radius_sc-0.5))

def optimisable_merit_function(x, radius_var_status, thickness_var_status, LDE, MFE):
    """A optimisable version of the merit function, takes new values within x
    and sets the parameters flagged as true in radius_var_status and thickness_var_status.
    x-Array: First Variable Radii in Order, then variable Thicknesses in order"""
    k = 0
    l = np.count_nonzero(radius_var_status)
    x_scale = np.zeros_like(x)
    x_scale[0:l] = rescale_radius_inverse(x[0:l],1)
    x_scale[l:] = rescale_thickness(x[l:],100.)
    
    NoS = LDE.NumberOfSurfaces-1
    
    for i in np.arange(NoS):
        obj = LDE.GetSurfaceAt(i)
        if radius_var_status[i]:
            obj.RadiusCell.DoubleValue = x_scale[k]
            k += 1
        if thickness_var_status[i]:
            obj.ThicknessCell.DoubleValue = x_scale[l]
            l +=1
    return calc_merit(MFE)



def opt_annealing(radius_var_status, thickness_var_status, LDE, MFE, steps = 100):
    """Starts a dual annealing optimization and sets the lens parameters to the found optimum"""
    n_var = np.count_nonzero(radius_var_status) + np.count_nonzero(thickness_var_status)
    lw = [0] * n_var
    up = [1] * n_var
    ret = dual_annealing(optimisable_merit_function, bounds=list(zip(lw, up)), args = (radius_var_status, thickness_var_status, LDE, MFE), maxiter = steps)
    dummy = optimisable_merit_function(ret.x, radius_var_status, thickness_var_status, LDE, MFE)
    return ret

class ZOSOptimisationproblem(ElementwiseProblem):

    def __init__(self, n_var, radius_var_status, thickness_var_status, LDE, MFE):
        super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=0.0, xu=1.0)
        self.radius_var_status = radius_var_status 
        self.thickness_var_status = thickness_var_status
        self.LDE = LDE
        self.MFE = MFE
    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = optimisable_merit_function(x, radius_var_status=self.radius_var_status, thickness_var_status=self.thickness_var_status, LDE= self.LDE, MFE= self.MFE)


def opt_nsga(radius_var_status, thickness_var_status, LDE, MFE, algo = "NSGA2"):
    """Starts a NSGA2/3 Optimization algorithm and sets the lens parameters to the found optimum"""
    n_var = np.count_nonzero(radius_var_status) + np.count_nonzero(thickness_var_status)
    problem = ZOSOptimisationproblem(n_var, radius_var_status, thickness_var_status, LDE, MFE)

    if algo == "NSGA2":
        algorithm = NSGA2(pop_size=100)
    elif algo == "NSGA3":
        ref_dirs = get_reference_directions("das-dennis", 1, n_partitions=12)
        algorithm = NSGA3(pop_size=100, ref_dirs = ref_dirs)
    else:
        raise ValueError("Select either 'NSGA2' or 'NSGA3'")


    res = minimize(problem,
                algorithm,
                ('n_gen', 200),
                seed=1,
                verbose=False)
    dummy = optimisable_merit_function(res.X, radius_var_status, thickness_var_status, LDE, MFE)
    return res

