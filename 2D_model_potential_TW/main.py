#this is a 2D potential, similar to the 1D_model_potential_TW\main.py
#Written by TW 23th May

from math import pi
from matplotlib import pyplot as plt
import numpy as np
from util_2d import *

#first we initialize some parameters.

N = 20 #number of grid points, i.e. num of states.
kT = 0.5981
t_max = 10**7 #max time
ts = 0.1 #time step

state_start = (1, 1)
state_end = (18, 18)

#initialize a 2d potential surface. default N=100
K = create_K_2D(N, kT)

#plot the K as countour
plt.contourf(K)
plt.show()

#get peq and F
peq, F, evectors, evalues, evalues_sorted, index = compute_free_energy(K, kT)

#plot the F as countour #x, y is 0 to 5pi
plt.contourf(F)
plt.xlim(0, 5*pi)
plt.ylim(0, 5*pi)
plt.show()


mfpt = mfpt_calc_2D(peq, K) #mfpt in shape (N,N,N,N) from state(i,j) to state(k,l) 

print("Unperturbed mfpt from rate matrix K", state_start," -> ", state_end, "MFPT:", mfpt[state_start[0], state_start[1], state_end[0], state_end[1]])

M = expm(K*ts) #transition matrix
Mmfpt = markov_mfpt_calc_2D(peq, M) * ts

print("Unperturbed mfpt from transition matrix M", state_start," -> ", state_end, "MFPT:", Mmfpt[state_start[0], state_start[1], state_end[0], state_end[1]])


##################
# now that we have constructed the initial system, let's perturb it.
##################






print("All done.")