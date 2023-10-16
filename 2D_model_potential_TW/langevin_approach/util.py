#utility file for langevin_sim_mfpt_opt.py
#bt TW 9th Oct 2023.
# Path: langevin_approach/util.py

import numpy as np
from scipy.linalg import eig
from scipy.linalg import expm
from scipy.linalg import inv
from scipy.optimize import minimize
from math import pi
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

def gaussian_2d(x, y, ax, bx, by, cx, cy):
    return ax * np.exp(-((x-bx)**2/(2*cx**2) + (y-by)**2/(2*cy**2)))

def random_initial_bias_2d(initial_position, num_gaussians = 20):
    # initial position is a list e.g. [3,3]
    # note this is in 
    #returns a set of random ax,ay, bx, by, cx, cy for the 2d Gaussian function
    rng = np.random.default_rng()
    a = np.ones(num_gaussians) * 0.1#* 4 #
    #ay = np.ones(num_gaussians) * 0.1 #there's only one amplitude!
    bx = rng.uniform(initial_position[0]-1, initial_position[0]+1, num_gaussians)
    by = rng.uniform(initial_position[1]-1, initial_position[1]+1, num_gaussians)
    cx = rng.uniform(1.0, 5.0, num_gaussians)
    cy = rng.uniform(1.0, 5.0, num_gaussians)
    return np.concatenate((a, bx, by, cx, cy))

def get_total_bias_2d(x,y, gaussian_params):
    """
    here we get the total bias at x,y.
    note: we used the transposed K matrix, we need to apply transposed total gaussian bias.
    """
    N = x.shape[0] #N is the number of grid points.
    total_bias = np.zeros((N,N))
    num_gaussians = len(gaussian_params)//5
    a = gaussian_params[:num_gaussians]
    bx = gaussian_params[num_gaussians:2*num_gaussians]
    by = gaussian_params[2*num_gaussians:3*num_gaussians]
    cx = gaussian_params[3*num_gaussians:4*num_gaussians]
    cy = gaussian_params[4*num_gaussians:5*num_gaussians]
    for i in range(num_gaussians):
        total_bias = total_bias + gaussian_2d(x, y, a[i], bx[i], by[i], cx[i], cy[i])

    return np.flipud(total_bias)

def compute_free_energy(K, kT):
    """
    In 2D senario, we just need to reshape the peq and F.

    K is the transition matrix
    kT is the thermal energy
    peq is the stationary distribution #note this was defined as pi in Simian's code.
    F is the free energy
    eigenvectors are the eigenvectors of K

    first we calculate the eigenvalues and eigenvectors of K
    then we use the eigenvalues to calculate the equilibrium distribution: peq.
    then we use the equilibrium distribution to calculate the free energy: F = -kT * ln(peq)
    """
    N = int(np.sqrt(K.shape[0]))
    evalues, evectors = eig(K)

    #sort the eigenvalues and eigenvectors
    index = np.argsort(evalues) #sort the eigenvalues, the largest eigenvalue is at the end of the list
    evalues_sorted = evalues[index] #sort the eigenvalues based on index

    #calculate the equilibrium distribution
    peq = evectors[:, index[-1]].T #normalize the eigenvector
    peq = peq / np.sum(peq)
    peq = peq.real
    #take the real part of the eigenvector i.e. the probability distribution at equilibrium.
    #calculate the free energy
    F = -kT * np.log(peq) #add a small number to avoid log(0))
    #F = F.reshape(N, N)
    return [peq, F, evectors, evalues, evalues_sorted, index]

def kemeny_constant_check(mfpt, peq):
    N2 = mfpt.shape[0]
    kemeny = np.zeros((N2, 1))
    for i in range(N2):
        for j in range(N2):
            kemeny[i] = kemeny[i] + mfpt[i, j] * peq[j]
    print("Performing Kemeny constant check...")
    print("the min/max of the Kemeny constant is:", np.min(kemeny), np.max(kemeny))
    """
    if np.max(kemeny) - np.min(kemeny) > 1e-6:
        print("Kemeny constant check failed!")
        raise ValueError("Kemeny constant check failed!")"""
    return kemeny

def mfpt_calc(peq, K):
    """
    peq is the probability distribution at equilibrium.
    K is the transition matrix.
    N is the number of states.
    """
    N = K.shape[0] #K is a square matrix.
    onevec = np.ones((N, 1)) #, dtype=np.float64
    Qinv = np.linalg.inv(peq.T * onevec - K.T)

    mfpt = np.zeros((N, N)) #, dtype=np.float64
    for j in range(N):
        for i in range(N):
            #to avoid devided by zero error:
            if peq[j] == 0:
                mfpt[i, j] = 0
            else:
                mfpt[i, j] = 1 / peq[j] * (Qinv[j, j] - Qinv[i, j])
    
    #result = kemeny_constant_check(N, mfpt, peq)
    return mfpt

def Markov_mfpt_calc(peq, M):
    N = M.shape[0]
    onevec = np.ones((N, 1))
    Idn = np.diag(onevec[:, 0])
    A = (peq.reshape(-1, 1)) @ onevec.T #was peq.T @ onevec.T
    A = A.T
    Qinv = inv(Idn + A - M)
    mfpt = np.zeros((N, N))
    for j in range(N):
        for i in range(N):
            term1 = Qinv[j, j] - Qinv[i, j] + Idn[i, j]
            if peq[j] * term1 == 0:
                mfpt[i, j] = 1000000000000
            else:
                mfpt[i, j] = 1/peq[j] * term1
    #result = kemeny_constant_check(N, mfpt, peq)
    return mfpt

def try_and_optim_M(M, working_indices, N=20, num_gaussian=10, start_index=0, end_index=0, plot = False):
    """
    here we try different gaussian params 1000 times
    and use the best one (lowest mfpt) to local optimise the gaussian_params
    
    returns the best gaussian params

    input:
    M: the working transition matrix, square matrix.
    working_indices: the indices of the working states.
    num_gaussian: number of gaussian functions to use.
    start_state: the starting state. note this has to be converted into the index space.
    end_state: the ending state. note this has to be converted into the index space.
    index_offset: the offset of the index space. e.g. if the truncated M (with shape [20, 20]) matrix starts from 13 to 33, then the index_offset is 13.
    """
    #here we find the index of working_indices.
    # e.g. the starting index in the working_indices is working_indices[start_state_working_index]
    # and the end is working_indices[end_state_working_index]
    
    N = N
    start_state_working_index = np.argmin(np.abs(working_indices - start_index))
    end_state_working_index = np.argmin(np.abs(working_indices - end_index))
    
    start_state_working_index_xy = np.unravel_index(working_indices[start_state_working_index], (N, N), order='C')
    end_state_working_index_xy = np.unravel_index(working_indices[end_state_working_index], (N, N), order='C')
    print("Try and Optim from state:", start_state_working_index_xy, "to state:", end_state_working_index_xy)

    #now our M/working_indices could be incontinues. #N = M.shape[0]
    x,y = np.meshgrid(np.linspace(-3,3, N), np.linspace(-3,3, N)) #hard coded here. we need to change this.
    best_mfpt = 1e12 #initialise the best mfpt np.inf

    #here we find the x,y maximum and minimun in xy coordinate space, with those working index
    #we use this to generate the random gaussian params.
    working_indices_xy = np.unravel_index(working_indices, (N, N), order='C')

    for try_num in range(1000):
        rng = np.random.default_rng()
        #a = rng.uniform(0.1, 1, num_gaussian)
        a = np.ones(num_gaussian)
        bx = rng.uniform(-3, 3, num_gaussian)
        by = rng.uniform(-3, 3, num_gaussian)
        #bx = rng.uniform(x_min, x_max, num_gaussian)
        #by = rng.uniform(y_min, y_max, num_gaussian)

        cx = rng.uniform(0.3, 1.5, num_gaussian)
        cy = rng.uniform(0.3, 1.5, num_gaussian)
        gaussian_params = np.concatenate((a, bx, by, cx, cy))

        total_bias = get_total_bias_2d(x,y, gaussian_params)
        M_biased = np.zeros_like(M)

        #we truncate the total_bias to the working index.
        working_bias = total_bias[working_indices_xy] #say M is in shape[51,51], working bias will be in [51] shape.

        #now we have a discontinues M matrix. we need to apply the bias to the working index.
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                M_biased[i,j] = M[i,j] * np.exp(-(working_bias[j] - working_bias[i]) / (2*0.5981))
                M_biased[j,i] = M[j,i] * np.exp((working_bias[i] - working_bias[j]) / (2*0.5981))
            M_biased[i,i] = M[i,i]
        #normalize M_biased
        #epsilon_offset = 1e-15
        M_biased = M_biased / (np.sum(M_biased, axis=0)[:, None] + 1e-15)

        #M_biased = M_biased.real
        #note our M_biased is in working index. M.shape = (num_working_states, num_working_states)
        [peq, F, evectors, evalues, evalues_sorted, index] = compute_free_energy(M_biased, kT=0.5981)
        
        mfpts_biased = Markov_mfpt_calc(peq, M_biased)
        mfpt_biased = mfpts_biased[start_state_working_index, end_state_working_index]

        if try_num % 100 == 0:
            kemeny_constant_check(mfpts_biased, peq)
            print("random try:", try_num, "mfpt:", mfpt_biased)
        if best_mfpt > mfpt_biased:
            best_mfpt = mfpt_biased
            best_params = gaussian_params

    print("best mfpt:", best_mfpt)
    
    #now we use the best params to local optimise the gaussian params

    def mfpt_helper(gaussian_params, M, start_state_working_index = start_state_working_index, end_state_working_index = end_state_working_index, kT=0.5981, working_indices=working_indices):
        #print("Try and Optim from state:", start_state_working_index_xy, "to state:", end_state_working_index_xy)
        total_bias = get_total_bias_2d(x,y, gaussian_params)
        M_biased = np.zeros_like(M)
        #we truncate the total_bias to the working index.
        working_bias = total_bias[working_indices_xy] #say M is in shape[51,51], working bias will be in [51] shape.

        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                M_biased[i,j] = M[i,j] * np.exp(-(working_bias[j] - working_bias[i]) / (2*0.5981))
                M_biased[j,i] = M[j,i] * np.exp((working_bias[i] - working_bias[j]) / (2*0.5981))
            M_biased[i,i] = M[i,i]
        #normalize M_biased
        #epsilon_offset = 1e-15
        M_biased = M_biased / (np.sum(M_biased, axis=0)[:, None] + 1e-15)
        #M_biased = M_biased.real
        #note our M_biased is in working index. M.shape = (num_working_states, num_working_states)
        [peq, F, evectors, evalues, evalues_sorted, index] = compute_free_energy(M_biased, kT=0.5981)
        
        mfpts_biased = Markov_mfpt_calc(peq, M_biased)
        mfpt_biased = mfpts_biased[start_state_working_index, end_state_working_index]
        return mfpt_biased

    
    res = minimize(mfpt_helper, 
                   best_params, 
                   args=(M,
                         start_state_working_index, 
                         end_state_working_index,
                         working_indices), 
                   method='Nelder-Mead', 
                   bounds= [(0.1, 3)]*num_gaussian + [(-3,3)]*num_gaussian + [(-3,3)]*num_gaussian + [(0.3, 1.5)]*num_gaussian + [(0.3, 1.5)]*num_gaussian,
                   tol=1e1)
    
    #print("local optimisation result:", res.x)
    return res.x 

def save_CV_total(CV_total, time_tag, prop_index):
    np.save(f"./data/{time_tag}_{prop_index}_CV_total.npy", CV_total[-1])

def save_gaussian_params(gaussian_params, time_tag, prop_index):
    np.save(f"./data/{time_tag}_{prop_index}_gaussian_params.npy", gaussian_params)



