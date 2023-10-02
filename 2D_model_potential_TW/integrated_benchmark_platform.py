#by TW 28th Sep
#this is an integrated benchmark platform for the whole project
# main benchmarking algorithm:
#  langevin dynamics (provided by Denes)
#  unbiased stocastic simulation (provided by TW)
#  K optimized (provided by TW)
#  exploring M optimized (provided by TW)
#  metadynamics (provided by Oisin)

import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm

#openmm for langevin dynamics

#self defined dependencies
from util_2d import *


#initialize the parameters
N = 20
kT = 0.5981
ts = 0.01 #time step
state_start = (14, 14)
state_end = (4, 7)

amp = 7
propagation_step = 1000
max_propagation = 50
num_bins = 20
num_gaussian = 20
num_simulations = 20


#suppose we wrap up all the algorithms in a function
# input is the algorithm name, args, and kwargs
# output is the mfpt (time_to_reach).


