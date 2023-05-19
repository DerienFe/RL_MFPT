#By Tiejun Wei 16th May 2023
#This file is the main file to run the 1D model potential code

from math import pi, sin
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from util import *

#first we initialize some parameters.

N = 100 #number of grid points, i.e. num of states.
force_constant = 0.1
kT = 0.5981

#create a K matrix for 1D model potential.
K = create_K_1D(N, kT)

"""#here we briefly plot K to see if it is correct
import matplotlib.pyplot as plt
plt.imshow(K)
plt.show()
"""

peq, F, evectors, evalues, evalues_sorted, index = compute_free_energy(K, kT)

#zero the free energy
F = F - np.min(F)

"""#plot the free energy
import matplotlib.pyplot as plt
plt.plot(F)
plt.show()"""

mfpt = mfpt_calc(peq, K, N)

#take assumption we start at state 9, end at state 89.
mfpt_9_89 = mfpt[8, 88]
print("Unperturbed. State 9 -> 89 MFPT: ", mfpt_9_89)
mfpt_9_89_opt = mfpt_9_89

##############################################################
#In the second step. We apply 5 random gaussian functions to the free energy.
#we repeat this process 1000 times and update the lowest mfpt.
##############################################################

#here we define a function wrap up the perturbation, K_bias, re-evaluating mfpt process.
#and we update the mftp everytimes we apply the perturbation.
def perturbation(K, F, kT, cutoff, N, mfpt_9_89_ref, iter_num):
    #setup random seed
    mfpt_9_89_opt = mfpt_9_89_ref #initialize the mfpt_opt
    np.random.seed(iter_num)

    #setup the gaussian functions
    def gaussian(x, a, b, c):
        return a * np.exp(-(x - b)**2 / ((2*c)**2)) #c is the "std_g" in Simian's code. b is the c_g.

    #setup the parameters for the gaussian functions
    a = [1]*5 # according to simian code
    b = np.random.rand(5) * 100
    c = np.random.rand(5) * 25

    #initialize the gaussian functions
    gaussian_functions = []
    for i in range(5):
        gaussian_functions.append(gaussian(np.linspace(0, 99, 100), a[i], b[i], c[i])) #5 gaussians start from 0 to 99.

    #we sum up all the random gaussian functions as total_bias
    total_bias = np.sum(gaussian_functions, axis=0)
    total_bias_opt = None

    #apply the 5 gaussian functions to the free energy and plot
    F_biased = F + total_bias

    """
    plot the biased free energy
    plt.plot(F, color='blue')
    plt.plot(F_biased, color='red')
    plt.plot(total_bias, color='green')
    plt.show()
    """
    #here we set the cutoff for the biased free energy
    cutoff = 20 #no unit.
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
    K_biased = bias_K(K, total_bias, kT, N=100, cutoff=20)

    #here we calculate the MFPT for the biased K matrix
    peq_biased, F_biased, evectors_biased, evalues_biased, evalues_sorted_biased, index_biased = compute_free_energy(K_biased, kT)
    mfpt_biased = mfpt_calc(peq_biased, K_biased, N)
    #print(mfpt_biased)
    #we update the mftp_opt if the new mfpt is smaller than the old one.
    #need to fix this. update all param if mfpt is smaller. otherwise, keep the original param.
    #have to fix the null value for params.
    print("Biasing attempt:", iter_num ," biased mfpt is: ", mfpt_biased[8, 88], "the mfpt_opt is: ", mfpt_9_89_opt)
    
    if mfpt_biased[8, 88] < mfpt_9_89_ref:
        mfpt_9_89_opt = mfpt_biased[8, 88]
        total_bias_opt = total_bias
    return [total_bias_opt, mfpt_9_89_opt]

    

#here we repeat the perturbation process 1000 times. MC style.
for i in range(100): #100iter: Final MFPT:  3801.16;;; 3000iter: Final MFPT:  2690.27
    result = perturbation(K, F, kT, 20, N, mfpt_9_89_opt,i)

    if result[0] is not None:
        total_bias_opt = result[0]
    mfpt_9_89_opt = result[1]

print("Final MFPT: ", mfpt_9_89_opt)
#note the optimial total_bias is stored in total_bias_opt now.

#plot the biased free energy
plt.plot(F, color='blue')
#plt.plot(F_biased, color='red')
#plt.plot(total_bias_opt, color='green')
plt.show()

##########################################
# Third step. We apply the previously calculated optimal bias to the K matrix.
##########################################

#here we use the optimal bias in the last Monte Carlo simulation.
#apply the opt bias, calculate K, MFPT.

K_biased_opt = bias_K(K, total_bias_opt, kT, N=100, cutoff=20)
peq_biased_opt = compute_free_energy(K_biased_opt, kT)[0] #only take peq here.
mfpt_biased_opt = mfpt_calc(peq_biased_opt, K_biased_opt, N=100)

#here we use another method. original the matlab code 'explore_the_network'

t_max = 10**7 #max time
ts = 0.01 #time step
num_simultation = 3 #number of simulations

state_start = 9
state_end = 89

state_visited = []
steps_needed = []

#here we random walk 3 times. record the sim_result.
for i in range(num_simultation):
    sim_result = explore_the_network(t_max, ts, K_biased_opt, state_start, state_end, N=100)
    state_visited.append(sim_result[0])
    steps_needed.append(sim_result[1])

print(np.mean(steps_needed)/100) # we take the mean of simulation steps. 

#plot the nodes visited.
plt.plot(state_visited[0], color='blue')
plt.plot(state_visited[1], color='red')
plt.plot(state_visited[2], color='green')
plt.show()


#here we unbias using DHAM.
#under construction.
x_eq = np.array(range(100)).reshape[100,1] #i dont understand.

prob_dist = DHAM_unbias(state_visited, x_eq, kT, N=100, bias = np.zeros([1, 100]), force_constant=0.1, cutoff=20)
F_DHAM = -kT * np.log(prob_dist) # calculate the free energy surface based on DHAM unbiasing.

#plot DHAM prob_dist
plt.plot(prob_dist, color='blue')
plt.show()
