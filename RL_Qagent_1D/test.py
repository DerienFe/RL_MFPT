#this is testing file.
from util import *
import matplotlib.pyplot as plt

N = 100
kT = 0.596
state_start = 7
state_end = 88

#initialize the K as state
K = create_K_1D(N, kT)

[peq, F, evectors, evalues, evalues_sorted, index] = compute_free_energy(K, kT)
F = F - min(F)
plt.plot(F)
plt.show()

mfpt = mfpt_calc(peq, K)

#take assumption we start at state 9, end at state 89.
mfpt_poi = mfpt[state_start, state_end] #point of interest mfpt.
print("Unperturbed ", state_start," -> ", state_end, "MFPT:", mfpt_poi)
mfpt_poi_opt = mfpt_poi #initialize the mfpt_poi_opt

gaussian_bias = gaussian(np.linspace(0, 100, 100), a=1, b=4, c=1)
bias_K = bias_K_1D(K, gaussian_bias)

print("all done")