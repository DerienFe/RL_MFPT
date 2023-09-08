from math import pi
from matplotlib import pyplot as plt
import numpy as np
from util_2d import *

#first we initialize some parameters.

N = 13 #number of grid points, i.e. num of states.
kT = 0.5981
t_max = 10**7 #max time
ts = 0.1 #time step

state_start = (11, 11)
state_end = (3, 11)


test = np.ravel_multi_index(state_start, (N,N), order='C')
test2 = np.unravel_index(test, (N,N), order='C')

K = create_K_2D(N, kT) 

#test the functions.
peq, F, evectors, evalues, evalues_sorted, index = compute_free_energy(K, kT)
mfpts = mfpt_calc(peq, K)
kemeny_constant_check(mfpts, peq)

#digitize the start end states.
state_start_index = np.ravel_multi_index(state_start, (N,N))

#plot the free energy surface
x,y = np.meshgrid(np.linspace(-3,3,N),np.linspace(-3,3,N))
plt.figure()
plt.contourf(x,y,F.reshape(N,N))#, levels=np.arange(0, 15, 0.5))
plt.colorbar()
plt.show()

from scipy.linalg import expm
M = expm(K*ts)

#plt.contourf(M)

#now we get the F_M out of M.
peq_M, F_M, evectors, evalues, evalues_sorted, index = compute_free_energy(M, kT)
mfpts = Markov_mfpt_calc(peq_M, M) * ts
kemeny_constant_check(mfpts, peq_M)

plt.figure()
plt.contourf(x,y,F_M.real.reshape(N,N))
plt.colorbar()
plt.show()


#test bias.
cur_pos = np.ravel_multi_index(state_start, (N,N), order='C')
gaussian_params = random_initial_bias_2d(initial_position = np.unravel_index(cur_pos, (N,N), order='C'), num_gaussians=10)
from main import get_total_bias_2d
total_bias = get_total_bias_2d(x,y, gaussian_params)
K_biased = bias_K_2D(K, total_bias)

peq, F, evectors, evalues, evalues_sorted, index = compute_free_energy(K_biased, kT)
mfpts = mfpt_calc(peq, K_biased)
kemeny_constant_check(mfpts, peq)
plt.figure()
plt.contourf(x,y,F.reshape(N,N))#, levels=np.arange(0, 15, 0.5))
plt.colorbar()
plt.show()



bias_M = bias_K_2D(M, total_bias)

peq_M, F_M, evectors, evalues, evalues_sorted, index = compute_free_energy(bias_M, kT)
mfpts = Markov_mfpt_calc(peq_M, bias_M)
kemeny_constant_check(mfpts, peq_M)
plt.figure()
plt.contourf(x,y,F_M.reshape(N,N))
plt.colorbar()
plt.show()

print("hello")


#we test random try here.
num_gaussian = 10
for i in range(1000):
    rng = np.random.default_rng()
    a = np.ones(num_gaussian)
    bx = rng.uniform(-3, 3, num_gaussian)
    by = rng.uniform(-3, 3, num_gaussian)
    cx = rng.uniform(1.0, 5.0, num_gaussian)
    cy = rng.uniform(1.0, 5.0, num_gaussian)
    gaussian_params = np.concatenate((a, bx, by, cx, cy))

    total_bias = get_total_bias_2d(x,y, gaussian_params)

    for i in range(M.shape[0]): #M here is NN shaped.
        for j in range(M.shape[1]):
            M[i,j] = M[i,j] * np.exp(-(total_bias[j] - total_bias[i]))

    