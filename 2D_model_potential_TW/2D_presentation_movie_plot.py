#a python script similar to main.py
#but we do it for presentation.

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
N = 20 #number of grid points, i.e. num of states.
kT = 0.5981
t_max = 10**7 #max time
ts = 0.001 #time step

state_start = (14, 14)
state_end = (4, 6)

#define 4 intermediate states between start and end.
state_1 = (10, 13)
state_2 = (8, 13)
state_3 = (6, 14)
state_4 = (6, 6)
intermediate_states = [state_1, state_2, state_3, state_4]
#time tag for saving.
time_tag = time.strftime("%Y%m%d-%H%M%S")

#for exploration.
propagation_step = 500
max_propagation = 10
num_bins = 20 #for qspace used in DHAM and etc.
num_gaussian = 20 #for the initial bias.

def propagate(M, cur_pos,
              gaussian_params,
              CV_total,prop_index, time_tag,
              steps=propagation_step, 
              stepsize=ts,):
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
        for intermediate_state in intermediate_states:
            intermediate_state = np.ravel_multi_index(intermediate_state, (N,N), order='C')
            if cur_pos == intermediate_state: #stop
                print(f"we have sampled the intermediate state pointat steps {i}")
                intermediate_states.pop(0)
                break
            #or in the ravelled 1D index we've passed the intermediate state, we stop.
            elif intermediate_state is not None and cur_pos < intermediate_state:
                print(f"we have passed the intermediate state pointat steps {i}")
                intermediate_states.pop(0)
                break

    combined_CV = np.concatenate((CV_total[-1], CV))
    CV_total.append(combined_CV)
    #plot where we have been, the CV space, in a heatmap, same size as the FES.
    #note this is in 2D, we unravel.
    x, y = np.meshgrid(np.linspace(-3,3,N),np.linspace(-3,3,N))
    pos = np.unravel_index(combined_CV.astype(int), (N,N), order='C')

    peq_M, F_M, evectors, evalues, evalues_sorted, index = compute_free_energy(M, kT)

    plt.figure()
    plt.imshow(F_M.reshape(N,N), cmap="coolwarm", extent=[-3,3,-3,3])#, levels=np.arange(0, 15, 0.5))
    #plot the traj in xy
    plt.plot(x[pos], -y[pos], color='black', markersize=0.3, linewidth=0.15, alpha=0.5)
    plt.plot(x[state_start], -y[state_start], marker='x')
    plt.plot(x[state_end], -y[state_end], marker='o')
    plt.savefig(f"./figs/{time_tag}_{prop_index}_traj.png")
    #plt.show()
    plt.close()

    
    #here we use the DHAM. #tobe done.
    F_M, MM = DHAM_it(combined_CV.reshape(-1,1), gaussian_params, T=300, lagtime=1, numbins=num_bins, prop_index=prop_index, time_tag=time_tag)
    
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

def DHAM_it(CV, gaussian_params, T=300, lagtime=2, numbins=150, prop_index=0, time_tag=time_tag):
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
    d.setup(CV, T, prop_index=prop_index, time_tag=time_tag)

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
    
    K = create_K_png(N)

    #test the functions.
    peq, F, evectors, evalues, evalues_sorted, index = compute_free_energy(K, kT)
    mfpts = mfpt_calc(peq, K)
    #kemeny_constant_check(mfpts, peq)

    x,y = np.meshgrid(np.linspace(-3,3,N),np.linspace(-3,3,N)) #for DHAM in 2D as well.
    
    #test random initial bias here.
    print("placing random gaussian at:", (x[state_start], y[state_start]))
    gaussian_params = random_initial_bias_2d(initial_position = [x[state_start], y[state_start]], num_gaussians=num_gaussian)
    total_bias = get_total_bias_2d(x,y, gaussian_params)
    K_biased = bias_K_2D(K, total_bias)
    peq_biased, F_biased, evectors_biased, evalues_biased, evalues_sorted_biased, index_biased = compute_free_energy(K_biased, kT)
    mfpts_biased = mfpt_calc(peq_biased, K_biased)
    #kemeny_constant_check(mfpts_biased, peq_biased)

    CV_total = [[]] #initialise the CV list.
    cur_pos = np.ravel_multi_index(state_start, (N,N), order='C') #flattened index.
    #note from now on, all index is in raveled 'flattened' form.
    for prop_index in range(max_propagation):
        if prop_index == 0:
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
                                                                prop_index=prop_index,
                                                                time_tag=time_tag,
                                                                steps=propagation_step, 
                                                                stepsize=ts,
                                                                )
            
            #our cur_pos is flattened 1D index.
            working_MM, working_indices = get_working_MM(M_reconstructed) #we call working_index the small index. its part of the full markov matrix.
            
            final_index = np.ravel_multi_index(state_end, (N,N), order='C') #flattened.

            #here we find the closest state to the final state.
            # first we unravel all the index to 2D.
            # then we use the lowest manhattan distance to find the closest state.
            # then we ravel it back to 1D.
            closest_index = find_closest_index(working_indices, final_index, N) 
        else:
            print(f"propagation number {prop_index} STARTING.")
            #renew the gaussian params using returned MM.

            #find the most visited state in CV_total[-1][-prop_step:]
            last_traj = np.array(CV_total[-1][-propagation_step:], dtype=int)
            most_visited_state = np.argmax(np.bincount(last_traj)) #this is in flattened index.
            most_visited_state_xy = np.unravel_index(most_visited_state, (N,N), order='C')
            
            gaussian_params = try_and_optim_M_K(working_MM, 
                                              working_indices = working_indices,
                                              num_gaussian=10, 
                                              start_index=cur_pos, 
                                              end_index=final_index,
                                              plot = True,
                                              )
            #save the gaussian_params.
            np.save(f"./data/{time_tag}_{prop_index}_gaussian_params.npy", gaussian_params)

            #renew the total bias.
            total_bias = get_total_bias_2d(x,y, gaussian_params)

                        #we get the FES biased.
            K_biased = bias_K_2D(K, total_bias)

            #for plot we need the detailed digitization fes.
            """ 
            img = Image.open("./fes_digitize.png")
            img = np.array(img)
            img_greyscale = 0.8 * img[:,:,0] - 0.15 * img[:,:,1] - 0.2 * img[:,:,2]
            img = img_greyscale
            img = img/np.max(img)
            img = img - np.min(img) #plt.imshow(img, cmap="coolwarm", extent=[-3,3,-3,3]) #note this is different than contourf.
            N_img = img.shape[0]
            K_img = img[:N_img, :N_img] * 7

            #we get x_img and y_img for gaussian to cast on.
            x_img,y_img = np.meshgrid(np.linspace(-3, 3, N_img), np.linspace(-3, 3, N_img))

            #now cast the gaussian on the x,y
            gaussian_img = get_total_bias_2d(x_img,y_img, gaussian_params)

            #add the gaussian to the img.

            #plot the F_img_biased.
            plt.figure()
            plt.imshow(gaussian_img, cmap="coolwarm", extent=[-3,3,-3,3])
            plt.title("detailed img_K biased")
            #set colorbar 0 to 12
            #plt.clim(0, 12)
            plt.colorbar()
            plt.show()"""


            peq_biased, F_biased, evectors_biased, evalues_biased, evalues_sorted_biased, index_biased = compute_free_energy(K_biased, kT)
            #we plot the total bias being applied on original FES.
            closest_index_xy = np.unravel_index(closest_index, (N,N), order='C')
            cur_pos_xy = np.unravel_index(cur_pos, (N,N), order='C')

            #plot the total bias. #the total bias is optimized in untransformed way.
            # so we don't need to transform it back. only those calculated FES need to be transformed back.

            plt.figure()
            plt.imshow(total_bias.reshape(N,N), cmap="coolwarm", extent=[-3,3,-3,3])
            plt.plot(x[state_start], -y[state_start], marker = 'o') #this is starting point.
            plt.plot(x[state_end], -y[state_end], marker = 'o') #this is ending point.
            plt.plot(x[closest_index_xy], -y[closest_index_xy], marker = 'v') #this is local run farest point.
            plt.plot(x[most_visited_state_xy], -y[most_visited_state_xy], marker = 'x') #this is local run farest point.
            plt.colorbar()
            plt.title(f"optimized total bias, prop_index = {prop_index}")
            plt.savefig(f"./figs/{time_tag}_{prop_index}_total_bias.png")
            plt.close()


            plt.figure()
            plt.imshow(F_biased.reshape(N,N), cmap="coolwarm", extent=[-3,3,-3,3])
            plt.plot(x[state_start], -y[state_start], marker = 'o') #this is starting point.
            plt.plot(x[state_end], -y[state_end], marker = 'o') #this is ending point.
            plt.plot(x[closest_index_xy], -y[closest_index_xy], marker = 'v') #this is local run farest point.
            plt.plot(x[most_visited_state_xy], -y[most_visited_state_xy], marker = 'x') #this is local run farest point.
            plt.colorbar()
            plt.title(f"total bias applied on FES, prop_index = {prop_index}")
            plt.savefig(f"./figs/{time_tag}_{prop_index}_fes_biased.png")
            plt.close()

            #apply the bias
            M = expm(K_biased*ts)
            #normalize M.
            for i in range(M.shape[0]): 
                M[:,i] = M[:,i]/np.sum(M[:,i])

            #we propagate the system again.
            F_M, cur_pos, M_reconstructed, CV_total  = propagate(M, cur_pos,
                                                                gaussian_params=gaussian_params, 
                                                                CV_total = CV_total,
                                                                prop_index=prop_index,
                                                                time_tag=time_tag,
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
            print(f"we have sampled the final state point, stop propagating at number {prop_index}")
            #here we plot the trajectory. The CV_total[-1]
            pos = np.unravel_index(CV_total[-1].astype(int), (N,N), order='C')

            #save the CV_total, for later statistics.
            save_CV_total(CV_total, time_tag, prop_index)
            break
        else:
            print("continue propagating.")
            continue