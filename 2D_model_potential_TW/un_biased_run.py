#this is a python script. use the 2D energy surface digitized from the png file and run
# unbiased stocastic simulation using np.random.choice to get the numerical mfpt from starting state to end state.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from tqdm import tqdm

from scipy.linalg import expm


from util_2d import *

#define few parameters.
N = 20
kT = 0.5981
ts = 0.01 #time step
state_start = (14, 14)
state_end = (4, 7)

num_simulations = 20

time_tag = time.strftime("%Y%m%d-%H%M%S")
if __name__ == "__main__":

    K = create_K_png(N)
    M = expm(K*ts)
    for i in range(M.shape[0]): 
        M[:,i] = M[:,i]/np.sum(M[:,i])


    cur_pos = np.ravel_multi_index(state_start, (N, N), order='C')
    final_pos = np.ravel_multi_index(state_end, (N, N), order='C')

    results = np.zeros(num_simulations)

    #we do this in a loop.
    for i_sum in range(num_simulations):
        cur_pos = np.ravel_multi_index(state_start, (N, N), order='C')
        final_pos = np.ravel_multi_index(state_end, (N, N), order='C')
        t = 0
        while cur_pos != final_pos:
            cur_pos = np.random.choice(np.arange(M.shape[0]), p=M[:,cur_pos])
            t += ts
        
        print("mfpt from state {} to state {} is {}".format(state_start, state_end, t))
        results[i_sum] = t
    #save the result in /data/

    #calculate the std and mean
    mean = np.mean(results)
    std = np.std(results)
    print("mean is {}, std is {}".format(mean, std))
    np.savetxt("./data/unbiased_mfpt_raw_{}_{}_{}.txt".format(state_start, state_end, time_tag), results)
    np.savetxt("./data/unbiased_mfpt_mean_std_{}_{}_{}.txt".format(state_start, state_end, time_tag), np.array([mean, std]))
        
        


    