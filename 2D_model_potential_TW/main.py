#here we use the "explore_bias_NaCl_gen.py" style, explor the FES of theoretical 2D system
#by TW 9th Aug.
from math import pi
from matplotlib import pyplot as plt
import numpy as np
from util_2d import *
from scipy.linalg import expm

import sys
import time
from tqdm import tqdm
from dham import DHAM
plt.rcParams.update({'font.size': 16})


#first we initialize some parameters.
N = 13 #number of grid points, i.e. num of states.
kT = 0.5981
t_max = 10**7 #max time
ts = 0.1 #time step

state_start = (3, 3)
state_end = (11, 11)

#for exploration.
propagation_step = 1000
max_propagation = 10
num_bins = 13 #for qspace used in DHAM and etc.
num_gaussian = 10 #for the initial bias.

def propagate(M, cur_pos,
              gaussian_params,
              CV_total,prop_index,
              steps=propagation_step, 
              stepsize=ts):
    """
    here we use the Markov matrix to propagate the system.
    save the CV and append it into the CV_total.
    use the DHAM_it to process CV_total, get the partially observed Markov matrix from trajectory.
    return the current position, the CV_total, and the partially observed Markov matrix.
    """
    N = int(np.sqrt(M.shape[0]))
    CV = [cur_pos]
    for i in tqdm(range(steps)):
        next_pos = int(np.random.choice(N*N, p = M[:, cur_pos]))
        CV.append(next_pos)
        cur_pos = next_pos

    combined_CV = np.concatenate((CV_total[-1], CV))
    CV_total.append(combined_CV)
    #plot where we have been, the CV space, in a heatmap, same size as the FES.
    #note this is in 2D, we unravel.
    pos = np.unravel_index(combined_CV.astype(int), (N,N), order='C')
    """
    plt.figure()
    plt.hist2d(pos[0], pos[1], bins=(N+1, N+1), cmap=plt.cm.jet)
    plt.plot(pos[0][0], pos[1][0], marker = 'x') #this is starting point.
    plt.plot(pos[0][-1], pos[1][-1], marker = 'o') #this is ending point.
    plt.colorbar()
    plt.xlim(0, N)
    plt.ylim(0, N)
    plt.show()
    """
    plt.figure()
    plt.plot(pos[0], pos[1], alpha=0.3)
    plt.plot(state_start[0], state_start[1], marker = 'x') #this is starting point.
    plt.plot(state_end[0], state_end[1], marker = 'o') #this is ending point.
    plt.xlim(0, N)
    plt.ylim(0, N)
    plt.show()
    
    
    #here we use the DHAM. #tobe done.
    F_M, MM = DHAM_it(combined_CV.reshape(-1,1), gaussian_params, T=300, lagtime=1, numbins=num_bins)
    
    return F_M, cur_pos, MM, CV_total


def get_working_MM(M):
    zero_rows = np.where(~M.any(axis=1))[0]
    zero_cols = np.where(~M.any(axis=0))[0]

    keep_indices = np.setdiff1d(range(M.shape[0]), np.union1d(zero_rows, zero_cols))
    M_work = M[np.ix_(keep_indices, keep_indices)]
    return M_work, keep_indices

def get_closest_state(qspace, target_state, working_indices):
    """
    usesage: qspace = np.linspace(2.4, 9, 150+1)
    target_state = 7 #find the closest state to 7A.
    """
    working_states = qspace[working_indices] #the NaCl distance of the working states.
    closest_state = working_states[np.argmin(np.abs(working_states - target_state))]
    return closest_state



def DHAM_it(CV, gaussian_params, T=300, lagtime=2, numbins=150):
    """
    intput:
    CV: the collective variable we are interested in. now it's 2d.
    gaussian_params: the parameters of bias potential. (in our case the 10-gaussian params)
     format: (a,bx, by,cx,cy)
    T: temperature 300

    output:
    the Markov Matrix
    Free energy surface probed by DHAM.
    """
    d = DHAM(gaussian_params)
    d.setup(CV, T)

    d.lagtime = lagtime
    d.numbins = numbins #num of bins, arbitrary.
    results = d.run(biased = True, plot=True)
    return results

def find_closest_index(working_indices, final_index, N):
    """
    returns the farest index in 1D.

    here we find the closest state to the final state.
    first we unravel all the index to 2D.
    then we use the lowest RMSD distance to find the closest state.
    then we ravel it back to 1D.
    note: for now we only find the first-encounted closest state.
          we can create a list of all the closest states, and then choose random one.
    """
    def rmsd_dist(a, b):
        return np.sqrt(np.sum((a-b)**2))
    working_x, working_y = np.unravel_index(working_indices, (N,N), order='C')
    working_states = np.stack((working_x, working_y), axis=1)
    final_state = np.unravel_index(final_index, (N,N), order='C')
    closest_state = working_states[0]
    for i in range(len(working_states)):
        if rmsd_dist(working_states[i], final_state) < rmsd_dist(closest_state, final_state):
            closest_state = working_states[i]
        
    closest_index = np.ravel_multi_index(closest_state, (N,N), order='C')
    return closest_index

###############################################
# here we start the main python process:
# propagate -> acess the Markov Matrix -> biasing -> propagate ...
###############################################

if __name__ == "__main__":
    
    K = create_K_2D(N, kT) 

    #test the functions.
    peq, F, evectors, evalues, evalues_sorted, index = compute_free_energy(K, kT)
    mfpts = mfpt_calc(peq, K)
    #kemeny_constant_check(mfpts, peq)

    x,y = np.meshgrid(np.linspace(-3,3,N),np.linspace(-3,3,N)) #for DHAM in 2D as well.
    
    #test random initial bias here.
    gaussian_params = random_initial_bias_2d(initial_position = [x[state_start], y[state_start]], num_gaussians=num_gaussian)
    total_bias = get_total_bias_2d(x,y, gaussian_params)
    K_biased = bias_K_2D(K, total_bias)
    peq_biased, F_biased, evectors_biased, evalues_biased, evalues_sorted_biased, index_biased = compute_free_energy(K_biased, kT)
    mfpts_biased = mfpt_calc(peq_biased, K_biased)
    #kemeny_constant_check(mfpts_biased, peq_biased)

    """    
    plt.figure()
    plt.contourf(x,y,(F-F.min()).reshape(13,13), levels=np.arange(0, 15, 0.5))
    plt.colorbar()
    plt.show()
    plt.figure()
    plt.contourf(x,y,(total_bias-total_bias.min()).reshape(13,13))
    plt.colorbar()
    plt.show()
    """

    CV_total = [[]] #initialise the CV list.
    cur_pos = np.ravel_multi_index(state_start, (N,N), order='C') #flattened index.
    #note from now on, all index is in raveled 'flattened' form.
    for i_prop in range(max_propagation):
        if i_prop == 0:
            print("propagation number 0 STARTING.")
            gaussian_params = random_initial_bias_2d(initial_position = np.unravel_index(cur_pos, (N,N), order='C'), num_gaussians=num_gaussian)
            total_bias = get_total_bias_2d(x,y, gaussian_params)
            K_biased = bias_K_2D(K, total_bias)

            #get Markov matrix.
            M = expm(K_biased*ts)
            #normalize M.
            for i in range(M.shape[0]): 
                M[:,i] = M[:,i]/np.sum(M[:,i])

            F_M, cur_pos, M_reconstructed, CV_total  = propagate(M, cur_pos,
                                                                gaussian_params=gaussian_params, 
                                                                CV_total = CV_total,
                                                                prop_index=i_prop,
                                                                steps=propagation_step, 
                                                                stepsize=ts)
            
            #our cur_pos is flattened 1D index.
            working_MM, working_indices = get_working_MM(M_reconstructed) #we call working_index the small index. its part of the full markov matrix.
            
            final_index = np.ravel_multi_index(state_end, (N,N), order='C') #flattened.

            #here we find the closest state to the final state.
            # first we unravel all the index to 2D.
            # then we use the lowest manhattan distance to find the closest state.
            # then we ravel it back to 1D.
            closest_index = find_closest_index(working_indices, final_index, N) 
        else:
            print(f"propagation number {i_prop} STARTING.")
            #renew the gaussian params using returned MM.

            gaussian_params = try_and_optim_M(working_MM, 
                                              working_indices = working_indices,
                                              num_gaussian=10, 
                                              start_index=cur_pos, 
                                              end_index=closest_index,
                                              plot = True,
                                              )

            #renew the total bias.
            total_bias = get_total_bias_2d(x,y, gaussian_params)

            #we get the FES biased.
            K_biased = bias_K_2D(K, total_bias)
            peq_biased, F_biased, evectors_biased, evalues_biased, evalues_sorted_biased, index_biased = compute_free_energy(K_biased, kT)
            #we plot the total bias being applied on original FES.
            closest_index_xy = np.unravel_index(closest_index, (N,N), order='C')
            cur_pos_xy = np.unravel_index(cur_pos, (N,N), order='C')
            plt.figure()
            plt.contourf(x,y,(F_biased-F_biased.min()).reshape(13,13))
            plt.plot(x[state_start], y[state_start], marker = 'x') #this is starting point.
            plt.plot(x[state_end], y[state_end], marker = 'o') #this is ending point.
            plt.plot(x[0][closest_index_xy[0]], y[closest_index_xy[1]][0], marker = 'v') #this is local run farest point.
            plt.plot()
            plt.colorbar()
            plt.savefig(f"./figs/fes_biased_{i_prop}.png")
            plt.show()

            #plot the total bias.
            plt.figure()
            plt.contourf(x,y,(total_bias - total_bias.min()).reshape(13,13))
            plt.plot(x[state_start], y[state_start], marker = 'x') #this is starting point.
            plt.plot(x[state_end], y[state_end], marker = 'o') #this is ending point.
            plt.plot(x[0][closest_index_xy[0]], y[closest_index_xy[1]][0], marker = 'v')
            plt.plot(x[0][cur_pos_xy[0]], y[cur_pos_xy[1]][0], marker = 'v') #this is local run current position.
            plt.colorbar()
            plt.savefig(f"./figs/total_bias_{i_prop}.png")
            plt.show()

            #apply the bias
            M = expm(K_biased*ts)
            #normalize M.
            for i in range(M.shape[0]): 
                M[:,i] = M[:,i]/np.sum(M[:,i])

            #we propagate the system again.
            F_M, cur_pos, M_reconstructed, CV_total  = propagate(M, cur_pos,
                                                                gaussian_params=gaussian_params, 
                                                                CV_total = CV_total,
                                                                prop_index=i_prop,
                                                                steps=propagation_step, 
                                                                stepsize=ts)
            
            #our cur_pos is flattened 1D index.
            working_MM, working_indices = get_working_MM(M_reconstructed) #we call working_index the small index. its part of the full markov matrix.
            
            
            #here we find the closest state to the final state.
            # first we unravel all the index to 2D.
            # then we use the lowest manhattan distance to find the closest state.
            # then we ravel it back to 1D.
            closest_index = find_closest_index(working_indices, final_index, N) 

        if closest_index == final_index:
            print(f"we have sampled the final state point, stop propagating at number {i_prop}")
            #here we plot the trajectory. The CV_total[-1]
            pos = np.unravel_index(CV_total[-1].astype(int), (N,N), order='C')
            plt.figure()
            plt.plot(pos[0], pos[1], alpha=0.3)
            plt.plot(state_start[0], state_start[1], marker = 'x') #this is starting point.
            plt.plot(state_end[0], state_end[1], marker = 'o') #this is ending point.
            plt.xlim(0, N)
            plt.ylim(0, N)
            plt.show()
            
            break
        else:
            print("continue propagating.")
            continue