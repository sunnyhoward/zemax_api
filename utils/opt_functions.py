'''
Optimisation functions...
'''

import numpy as np
import matplotlib.pyplot as plt

import os 
import sys
import h5py


def calculate_grad(Y,X):
    '''
    To be used with very simple gradient descent algorithm. 
    '''
    return (Y[1] - Y[0])/(X[1] - X[0])
