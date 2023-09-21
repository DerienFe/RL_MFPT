from math import pi
from matplotlib import pyplot as plt
import numpy as np
from util_2d import *
plt.rcParams.update({'font.size': 16})

#first we initialize some parameters.

N = 20 #number of grid points, i.e. num of states.
kT = 0.5981
t_max = 10**7 #max time
ts = 0.1 #time step

x,y = np.meshgrid(np.linspace(-3,3,N),np.linspace(-3,3,N))
state_start = (14, 14)
state_end = (4, 6)
#map it into xy mesh.

#test = np.ravel_multi_index(state_start_state, (N,N), order='C')
#test2 = np.unravel_index(test, (N,N), order='C')

K = create_K_png(N, kT) 

#test the functions.
peq, F, evectors, evalues, evalues_sorted, index = compute_free_energy(K, kT)
mfpts = mfpt_calc(peq, K)
kemeny_constant_check(mfpts, peq)

#digitize the start end states.
state_start_index = np.ravel_multi_index(state_start, (N,N), order='C')
state_end_index = np.ravel_multi_index(state_end, (N,N), order='C')


#plot the free energy surface

plt.figure()
plt.imshow(F.reshape(N,N), cmap="coolwarm", extent=[-3,3,-3,3])#, levels=np.arange(0, 15, 0.5))
plt.colorbar()
plt.title("unbiased fes digitized from Dialanine")
#plt.savefig("./figs/unbiased.png")
plt.show()

from scipy.linalg import expm
M = expm(K*ts)


#now we get the F_M out of M.
peq_M, F_M, evectors, evalues, evalues_sorted, index = compute_free_energy(M, kT)
mfpts = Markov_mfpt_calc(peq_M, M) * ts
kemeny_constant_check(mfpts, peq_M)

plt.figure()
plt.imshow(F_M.reshape(N,N), cmap="coolwarm", extent=[-3,3,-3,3])#, levels=np.arange(0, 15, 0.5))
plt.colorbar()
plt.show()


#test bias.
cur_pos = np.ravel_multi_index(state_start, (N,N), order='C')
#gaussian_params = random_initial_bias_2d(initial_position = np.unravel_index(cur_pos, (N,N), order='C'), num_gaussians=10)
gaussian_params = np.array([5, 1.51, -1.51, 1, 1])
from main import get_total_bias_2d
total_bias = get_total_bias_2d(x,y, gaussian_params)

K_biased = bias_K_2D(K, total_bias)

peq, F, evectors, evalues, evalues_sorted, index = compute_free_energy(K_biased, kT)
mfpts = mfpt_calc(peq, K_biased)
kemeny_constant_check(mfpts, peq)

plt.figure()
plt.imshow(F.reshape(N,N), cmap="coolwarm", extent=[-3,3,-3,3])#, levels=np.arange(0, 15, 0.5))
plt.plot(1.51, -1.51, marker = 'o', color = "red", markersize = 10) #this is starting point.
plt.colorbar()
plt.show()



M_biased = bias_K_2D(M, total_bias, norm=False)

#normalize the M_biased
M_biased = M_biased / np.sum(M_biased, axis=0)[:, np.newaxis]


#plt.contourf(M)
peq_M, F_M, evectors, evalues, evalues_sorted, index = compute_free_energy(M_biased, kT)
mfpts = Markov_mfpt_calc(peq_M, M_biased)
#kemeny_constant_check(mfpts, peq_M) #minor bug for MSM compute free energy. kemeny not zero.

print(mfpts[state_start_index, state_end_index])
plt.figure()
plt.imshow(F_M.reshape(N,N), cmap="coolwarm", extent=[-3,3,-3,3])#, levels=np.arange(0, 15, 0.5))
plt.colorbar()
plt.show()

print("hello")
############################################
best_mfpt = 1e8
num_gaussian = 10
for i in range(300):
    rng = np.random.default_rng()
    a = np.ones(num_gaussian) * 1
    bx = rng.uniform(-3, 3, num_gaussian)
    by = rng.uniform(-3, 3, num_gaussian)
    cx = rng.uniform(1.0, 5.0, num_gaussian)
    cy = rng.uniform(1.0, 5.0, num_gaussian)
    gaussian_params = np.concatenate((a, bx, by, cx, cy))

    total_bias = get_total_bias_2d(x,y, gaussian_params)

    M_biased = bias_K_2D(M, total_bias)

    #normalize the M_biased
    M_biased = M_biased / (np.sum(M_biased, axis=0)[:, None] + 1e-15)


    peq, F, evectors, evalues, evalues_sorted, index = compute_free_energy(M_biased, kT)
    mfpts = mfpt_calc(peq, K_biased)
    kemeny_constant_check(mfpts, peq)
    mfpt = mfpts[state_start_index, state_end_index]
    if mfpt < best_mfpt:
        best_mfpt = mfpt
        best_params = gaussian_params
        best_F = F
        best_K = K_biased
    
    print(i, mfpt, best_mfpt)

#examine the best one.
total_bias = get_total_bias_2d(x,y, best_params)
M_biased = bias_K_2D(M, total_bias, norm=False)
peq, F, evectors, evalues, evalues_sorted, index = compute_free_energy(M_biased, kT)
plt.figure()
plt.imshow(total_bias.reshape(N,N), cmap="coolwarm", extent=[-3,3,-3,3])#, levels=np.arange(0, 15, 0.5))
plt.title("optimized bias to minimize MFPT")
plt.colorbar()
plt.show()

#we test random try here.
best_mfpt = 1e8
num_gaussian = 10
for i in range(300):
    rng = np.random.default_rng()
    a = np.ones(num_gaussian) * 1
    bx = rng.uniform(-3, 3, num_gaussian)
    by = rng.uniform(-3, 3, num_gaussian)
    cx = rng.uniform(1.0, 5.0, num_gaussian)
    cy = rng.uniform(1.0, 5.0, num_gaussian)
    gaussian_params = np.concatenate((a, bx, by, cx, cy))

    total_bias = get_total_bias_2d(x,y, gaussian_params)

    K_biased = bias_K_2D(K, total_bias)

    peq, F, evectors, evalues, evalues_sorted, index = compute_free_energy(K_biased, kT)
    mfpts = mfpt_calc(peq, K_biased)
    #kemeny_constant_check(mfpts, peq)
    mfpt = mfpts[state_start_index, state_end_index]
    if mfpt < best_mfpt:
        best_mfpt = mfpt
        best_params = gaussian_params
        best_F = F
        best_K = K_biased
    
    print(i, mfpt, best_mfpt)

#examine the best one.
total_bias = get_total_bias_2d(x,y, best_params)
K_biased = bias_K_2D(K, total_bias)
peq, F, evectors, evalues, evalues_sorted, index = compute_free_energy(K_biased, kT)

plt.figure()
plt.imshow(F.reshape(N,N), cmap="coolwarm", extent=[-3,3,-3,3])#, levels=np.arange(0, 15, 0.5))
plt.title("biased_fes to minimize MFPT")
plt.colorbar()
plt.savefig("./figs/biased_fes.png")
plt.show()

plt.figure()
plt.contourf(x,y,total_bias, cmap="coolwarm", levels=100)#, levels=np.arange(0, 15, 0.5))
plt.title("optimized bias to minimize MFPT")
plt.colorbar()
plt.savefig("./figs/optimized_bias.png")
plt.show()

print("hello")


    