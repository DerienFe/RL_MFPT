#By Tiejun Wei 16th May 2023
#This file is the main file to run the 1D model potential code
#some persisting error, try state 50 -> 0. looks like from state 100 -> 50.
#check the mfpt functions, the matrix may need to be transposed.

from math import pi
from matplotlib import pyplot as plt
import numpy as np
from util import *

#first we initialize some parameters.

N = 100 #number of grid points, i.e. num of states.
kT = 0.5981
t_max = 10**7 #max time
ts = 0.1 #time step
num_simultation = 3 #number of simulations
num_gaussian = 10 #number of gaussian functions
state_start = 7 #note in this code this is 0-indexed. so 8 means state 9.
state_end = 88 #same as above. 88 means state 89.

#create a K matrix for 1D model potential.
K = create_K_1D(N, kT)

peq, F, evectors, evalues, evalues_sorted, index = compute_free_energy(K, kT)

#zero the free energy
F = F - np.min(F)

"""#plot the free energy
import matplotlib.pyplot as plt
plt.plot(F)
plt.show()"""

mfpt = mfpt_calc(peq, K)

#take assumption we start at state 9, end at state 89.
mfpt_poi = mfpt[state_start, state_end] #point of interest mfpt.
print("Unperturbed ", state_start," -> ", state_end, "MFPT:", mfpt_poi)
mfpt_poi_opt = mfpt_poi #initialize the mfpt_poi_opt

#check = kemeny_constant_check(N, mfpt, peq)


##############################################################
#In the second step. We apply 5 random gaussian functions to the free energy.
#we repeat this process 1000 times and update the lowest mfpt.
##############################################################

#here we define a function wrap up the perturbation, K_bias, re-evaluating mfpt process.
#and we update the mftp everytimes we apply the perturbation.
def perturbation(K, kT, N, mfpt_poi_ref, iter_num, num_gaussian=num_gaussian):
    #setup random seed
    mfpt_poi_opt = mfpt_poi_ref #initialize the _opt values
    abc_opt = None
    total_bias_opt = None
    #np.random.seed(iter_num) #set random seed for reproducibility.

    #setup the parameters for the gaussian functions
    a = np.random.rand(num_gaussian) * 1
    b = np.random.rand(num_gaussian) * 100
    c = np.random.rand(num_gaussian) * 25

    #initialize the gaussian functions
    gaussian_functions = []
    for i in range(num_gaussian):
        gaussian_functions.append(gaussian(np.linspace(0, 99, 100), a[i], b[i], c[i])) #5 gaussians start from 0 to 99.

    #we sum up all the random gaussian functions as total_bias
    total_bias = np.sum(gaussian_functions, axis=0)

    #apply the 5 gaussian functions to the free energy and plot
    #F_biased = F + total_bias

    """
    plot the biased free energy
    plt.plot(F, color='blue')
    plt.plot(F_biased, color='red')
    plt.plot(total_bias, color='green')
    plt.show()
    """
    #here we set the cutoff for the biased free energy
    #cutoff = 20 #no unit.
    """
    K_biased = np.zeros([N, N])
    for i in range(N):
        u_ij = total_bias - total_bias[i]  # Calculate u_ij (Note: Indexing starts from 0)
        u_ij[u_ij > cutoff] = cutoff  # Apply cutoff to u_ij
        u_ij[u_ij < -cutoff] = -cutoff

        KK = K[i, :]
        KK = KK.T * np.exp(u_ij / (2 * kT))  # Update KK

        K_biased[i, :] = KK.flatten()  # Assign KK to K_biased
        K_biased[i, i] = 0  # Set diagonal element to 0

    #normalizing the biased K matrix.
    for i in range(N):
        K_biased[i,i] = -np.sum(K_biased[:,i])
    """
    K_biased = bias_K_1D(K, total_bias, kT, N=100, cutoff=20)

    #here we calculate the MFPT for the biased K matrix
    peq_biased, F_biased, evectors_biased, evalues_biased, evalues_sorted_biased, index_biased = compute_free_energy(K_biased, kT)
    #mfpt_biased = mfpt_calc(peq_biased, K_biased) #not using this.

    M_t = expm(K_biased*ts)
    Mmfpt_biased = markov_mfpt_calc(peq_biased.T, M_t) * ts  # we use markov probagation as updating rule.

    #print(mfpt_biased)
    #we update the mftp_opt if the new mfpt is smaller than the old one.
    #need to fix this. update all param if mfpt is smaller. otherwise, keep the original param.
    #have to fix the null value for params.
    print("Biasing attempt {}: biased mfpt: {:.2f} ; mfpt_opt: {:.2f}".format(iter_num, Mmfpt_biased[state_start, state_end], mfpt_poi_opt))

    if Mmfpt_biased[state_start, state_end] < mfpt_poi_ref:
        mfpt_poi_opt = Mmfpt_biased[state_start, state_end]
        total_bias_opt = total_bias
        abc_opt = np.array([a, b, c])#we stack a, b, c together as abc_opt return for later scipy optimization.
    return [total_bias_opt, abc_opt, mfpt_poi_opt]


#here we repeat the perturbation process 1000 times. MC style.
for i in range(1000): #100iter: Final MFPT:  3801.16;;; 3000iter: Final MFPT:  2690.27
    result = perturbation(K, kT, N, mfpt_poi_opt,i)

    if result[0] is not None:
        total_bias_opt = result[0]
        abc_opt = result[1]
    mfpt_poi_opt = result[2]

F_biased_opt = (F + total_bias_opt) - np.min(F + total_bias_opt) #we zero the free energy again.

print("Final MFPT: ", mfpt_poi_opt)
#note the optimial total_bias is stored in total_bias_opt now.

#plot the biased free energy
plt.plot(F, color='blue')
plt.plot(F_biased_opt, color='green', linestyle='--')
plt.plot(total_bias_opt, color='red')
plt.title("Free Energy with optimal bias via MC")
plt.show()

##########################################
# Third step. We apply the previously calculated optimal bias to the K matrix.
##########################################

#here we use the optimal bias in the last Monte Carlo simulation.
#apply the opt bias, calculate K, MFPT.

K_biased_opt = bias_K_1D(K, total_bias_opt, kT, N=100, cutoff=20)
peq_biased_opt = compute_free_energy(K_biased_opt, kT)[0] #only take peq here.
#moved to the bottom for comparison. #mfpt_biased_opt = mfpt_calc(peq_biased_opt, K_biased_opt)

#here we use another method. original the matlab code 'explore_the_network'

state_visited = []
steps_needed = []

#here we random walk 3 times. record the sim_result.
for i in range(num_simultation):
    print("starting stochastic simulation: ", i, "...")
    sim_result = explore_the_network(t_max, ts, K_biased_opt, state_start, state_end)
    state_visited.append(sim_result[0])
    steps_needed.append(sim_result[1])

avg_mfpt_stochastic = np.mean(steps_needed)*ts #we take the mean of simulation steps.
error_mfpt_stochastic = np.std(steps_needed)*ts #we take the mean of simulation steps.

print("#############################################")
print('Random Walk result:', np.mean(steps_needed)*ts) # we take the mean of simulation steps. 

#here we use two methods.
#1. use the K matrix to calculate MFPT.
#2. use the markov_mfpt_calc(peq_biased, M_t) to calculate MFPT.

#here we calculate the MFPT using the K matrix.
mfpt_from_K = mfpt_calc(peq_biased_opt, K_biased_opt)

#here we calculate the MFPT using the markov_mfpt_calc(peq_biased, M_t)
M_t = expm(K_biased_opt*ts)
Mmfpt = markov_mfpt_calc(peq_biased_opt.T, M_t) * ts

print("MFPT from rate matrix K: ", mfpt_from_K[8, 88])
print("MFPT from markov_mfpt_calc: ", Mmfpt[8, 88])

#############################
# here we use scipy.optimize to find the optimal bias. given initial a, b, c.
# the target function is 
#############################

from scipy.optimize import minimize

def wrapper_function(x, K, num_gaussian, kT, ts, state_start, state_end):
    """
    This is a helper function wraps up the min_mfpt function.
    """
    a = x[:num_gaussian]
    b = x[num_gaussian:2*num_gaussian]
    c = x[2*num_gaussian:3*num_gaussian]
    return min_mfpt([a, b, c], K, num_gaussian, kT, ts, state_start, state_end)

#abc_init = abc_opt #initial guess from MC
options = {'maxiter': 1e3}
bounds = [(0, None)] * (3 * num_gaussian) #set bound to be positive.
scipy_result = minimize(wrapper_function, abc_opt, args=(K, num_gaussian, kT, ts, state_start, state_end), method='Nelder-Mead', options = options, bounds=bounds)
optimized_params = scipy_result.x
a, b, c = optimized_params[:num_gaussian], optimized_params[num_gaussian:2*num_gaussian], optimized_params[2*num_gaussian:3*num_gaussian]
print("Optimal a, b, c by scipy: ", a, b, c)

#plot the optimal free energy

gaussian_functions = []
for i in range(num_gaussian):
    gaussian_functions.append(gaussian(np.linspace(0, 99, 100), a[i], b[i], c[i])) #5 gaussians start from 0 to 99.
total_bias_opt_scipy = np.sum(gaussian_functions, axis=0)

F_biased_opt_scipy = (F + total_bias_opt_scipy) - np.min(F + total_bias_opt_scipy) #we zero the free energy again.

plt.plot(F, color='blue')
plt.plot(F_biased_opt_scipy, color='green', linestyle='--')
plt.plot(total_bias_opt_scipy, color='red')
plt.title("Free Energy with optimal bias via scipy")
plt.show()

print("All Done.")
