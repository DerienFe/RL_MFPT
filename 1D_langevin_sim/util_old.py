#util file for applying mfpt method on 1-d NaCl system.
#by TW 26th July 2023

import numpy as np
from scipy.linalg import logm, expm
from scipy.optimize import minimize

from scipy.linalg import inv
from scipy.linalg import eig
import matplotlib.pyplot as plt
import sys 
import openmm
import config
from scipy.sparse import diags, eye
from scipy.sparse import linalg,diags
from numba import jit,njit

def gaussian(x, a, b, c): #self-defined gaussian function
        return a * np.exp(-(x - b)**2 / ((2*c)**2)) 

def create_K_1D(fes, N=200, kT=0.5981):
    #create the K matrix for 1D model potential
    #K is a N*N matrix, representing the transition rate between states
    #The diagonal elements are the summation of the other elements in the same row, i.e. the overall outflow rate from state i
    #The off-diagonal elements are the transition rate from state i to state j (or from j to i???)
    
    #input:
    #fes: the free energy profile, a 1D array.

    K = np.zeros((N,N), dtype=np.float64) #, dtype=np.float64
    for i in range(N-1):
        K[i, i + 1] = np.exp((fes[i+1] - fes[i]) / 2 / kT)
        K[i + 1, i] = np.exp((fes[i] - fes[i+1]) / 2 / kT)
    for i in range(N):
        K[i, i] = 0
        K[i, i] = -np.sum(K[:, i])
    return K

def kemeny_constant_check(N, mfpt, peq):
    kemeny = np.zeros((N, 1))
    for i in range(N):
        for j in range(N):
            kemeny[i] = kemeny[i] + mfpt[i, j] * peq[j]
    print("Performing Kemeny constant check...")
    print("the min/max of the Kemeny constant is:", np.min(kemeny), np.max(kemeny))
    """
    if np.max(kemeny) - np.min(kemeny) > 1e-6:
        print("Kemeny constant check failed!")
        raise ValueError("Kemeny constant check failed!")"""
    return kemeny

#define a function calculating the mean first passage time
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


def bias_K_1D(K, total_bias, kT=0.5981):
    """
    K is the unperturbed transition matrix.
    total_bias is the total biasing potential.
    kT is the thermal energy.
    This function returns the perturbed transition matrix K_biased.
    """
    N = K.shape[0]
    K_biased = np.zeros([N, N])#, #dtype=np.float64)

    for i in range(N-1):
        u_ij = total_bias[i+1] - total_bias[i]  # Calculate u_ij (Note: Indexing starts from 0)
        K_biased[i, i+1] = K[i, i+1] * np.exp(u_ij /(2*kT))  # Calculate K_biased
        K_biased[i+1, i] = K[i+1, i] * np.exp(-u_ij /(2*kT))

    for i in range(N):
        K_biased[i,i] = -np.sum(K_biased[:,i])
    return K_biased

def compute_free_energy(K, kT=0.5981):
    """
    K is the transition matrix
    kT is the thermal energy
    peq is the stationary distribution #note this was defined as pi in Simian's code.
    F is the free energy
    eigenvectors are the eigenvectors of K

    first we calculate the eigenvalues and eigenvectors of K
    then we use the eigenvalues to calculate the equilibrium distribution: peq.
    then we use the equilibrium distribution to calculate the free energy: F = -kT * ln(peq)
    """
    evalues, evectors = eig(K)

    #sort the eigenvalues and eigenvectors
    index = np.argsort(evalues) #sort the eigenvalues, the largest eigenvalue is at the end of the list
    evalues_sorted = evalues[index] #sort the eigenvalues based on index

    #calculate the equilibrium distribution
    peq = evectors[:, index[-1]].T #normalize the eigenvector
    peq = peq / np.sum(peq)
    peq = peq.real

    F = -kT * np.log(peq) #add a small number to avoid log(0)) # + 1e-16

    return [peq, F, evectors, evalues, evalues_sorted, index]

def compute_free_energy_power_method_sparse(K,kT = 0.5981):
    """
    Uses the power method to calculate the equilibrium distribution. num_iter is the number of iterations. This has been modified to work on sparse matrices.
    """
    num_iter = 1000
    N = K.shape[0]
    peq = np.ones(N)/N #set all states to have equal occupation probability.
    for i in range(num_iter):
        peq = K.dot(peq)
        peq /=np.sum(peq)
    F = -kT * np.log(peq)
    return [peq,F]

def compute_free_energy_power_method(K, kT=0.5981):
    """
    this use the power method to calculate the equilibrium distribution.
    num_iter is the number of iterations.
    """
    num_iter = 1000
    N = K.shape[0]
    peq = np.ones(N) / N #initialise the peq
    for i in range(num_iter):
        peq = np.dot(peq, K)
        peq = peq / np.sum(peq)
    F = -kT * np.log(peq)
    return [peq, F]

def bias_M_1D(M, total_bias, kT=0.5981):
    """
    M is the unperturbed transition matrix.
    total_bias is the total biasing potential.
    kT is the thermal energy.
    This function returns the perturbed transition matrix M_biased.
    """
    N = M.shape[0]
    M_biased = np.zeros([N, N])#, #dtype=np.float64)

    for i in range(N):
        for j in range(N):
            u_ij = total_bias[j] - total_bias[i]
            M_biased[i, j] = M[i, j] * np.exp(-u_ij / (2*kT))
        M_biased[i, i] = M[i,i]

    """for i in range(N):
        if np.sum(M_biased[:, i]) != 0:
            M_biased[:, i] = M_biased[:, i] / np.sum(M_biased[:, i])
        else:
            M_biased[:, i] = 0"""
    #M_biased = M_biased/(np.sum(M_biased, axis=1)[:,None])# + 1e-15)
    for i in range(M_biased.shape[0]):
        row_sum = np.sum(M_biased[i, :])
        if row_sum > 0:
            M_biased[i, :] = M_biased[i, :] / row_sum
        else:
            M_biased[i, :] = 0
    return M_biased.real

#below Markov_mfpt_calc is provided by Sam M.
def Markov_mfpt_calc_sparse(peq, M): ##### implement jj hunter algorithm
    N = M.shape[0]
    onevec = np.ones(N)
    Idn = eye(N)  # Sparse identity matrix
    # print("peq: ",peq)
    # print("peq reshaped (-1,1)",peq.reshape(-1,1))
    peq_new = config.csc_matrix(peq.reshape(-1, 1))
    A = (peq_new.multiply(onevec.T)).T
    #print(M.data)
    #print(Idn + A - M)
    Qinv = linalg.inv((Idn + A - M).astype('float64'))  # Sparse matrix inversion
    Qinv_diag = config.csr_matrix(np.tile(Qinv.diagonal(),(Qinv.shape[0],1)))
    term = Qinv_diag - Qinv +Idn
    peq_row = peq[None:]
    term_array = term.toarray()
    mfpt = np.where((term_array != 0) & (peq_row != 0), 1 / peq_row * term_array, 1000000000000)
    #mfpt_optimized_sparse = config.sparse_mat[1](mfpt_dense)
    return mfpt

def Markov_mfpt_calc(peq, M):
    N = M.shape[0]
    onevec = np.ones((N, 1))
    Idn = np.diag(onevec[:, 0])
    #print("peq is:",peq)
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

##### used
def try_and_optim_sparse_M(M,working_indices, num_gaussian=10, start_index=0, end_index=0, plot = False):
    #print("inside try and optim_M")
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
    x = np.linspace(0, 2*np.pi, config.num_bins+1) #hard coded for now.
    best_mfpt = 1e20 #initialise the best mfpt np.inf

    #first we convert the big index into "index to the working indices".

    #start_state_working_index = np.where(working_indices == start_state)[0][0] #convert start_state to the offset index space.
    #end_state_working_index = np.where(working_indices == end_state)[0][0] #convert end_state to the offset index space.
    start_state_working_index = np.argmin(np.abs(working_indices - start_index))
    end_state_working_index = np.argmin(np.abs(working_indices - end_index))
    
    print("optimizing to get g_param from start state:", start_state_working_index, "to end state:", end_state_working_index, "in working indices.")
    print("converted to xspace that's from:", x[working_indices[start_state_working_index]], "to", x[working_indices[end_state_working_index]])
    
    #now our M/working_indices could be incontinues. #N = M.shape[0]
    
    #we get the upper/lower bound of the gaussian params.
    upper = x[working_indices[-1]]
    lower = x[working_indices[0]]
    print("upper bound:", upper, "lower bound:", lower)

    for try_num in range(1000): 
        rng = np.random.default_rng()
        #we set a to be 1
        a = np.ones(num_gaussian) *0.35
        b = rng.uniform(0, 2*np.pi, num_gaussian)
        #b = rng.uniform(lower, upper, num_gaussian)
        c = rng.uniform(0.3, 2, num_gaussian)
        
        #we convert the working_indices to the qspace.

        total_bias = np.zeros_like(x)
        for j in range(num_gaussian):
            total_bias += gaussian(x, a[j], b[j], c[j])

        #now we need to convert the total_bias to the working_indices space.
        #working_bias = total_bias[working_indices]
        
        #M_biased = bias_M_1D(M, total_bias, kT=0.5981)
        N = M.shape[0]
        data = []
        indices = []
        indptr = [0]

        # Iterate over non-zero elements of M to fill data, indices, and indptr
        for i in range(N):
            for j_indptr in range(M.indptr[i], M.indptr[i + 1]):
                j = M.indices[j_indptr]
                u_ij = total_bias[j] - total_bias[i]
                new_value = M.data[j_indptr] * np.exp(-u_ij / (2 * 0.5981))
                data.append(new_value)
                indices.append(j)
            indptr.append(len(data))

        # Create a new CSR matrix M_biased from data, indices, and indptr
        M_biased = config.sparse_mat[1]((data, indices, indptr), shape=(N, N))
        #print(M_biased)
        #peq,F,_,_,_,_  = compute_free_energy(M_biased.T.astype(np.float64), kT=0.5981)
        peq, F = compute_free_energy_power_method_sparse(M_biased, kT=0.5981)
        #return peq, M_biased
        mfpts_biased = Markov_mfpt_calc_sparse(peq, M_biased)
        mfpt_biased = mfpts_biased[start_index, end_index]
        #print(peq)
        #kemeny_constant_check(M.shape[0], mfpts_biased, peq)
        if try_num % 100 == 0:
            print("random try:", try_num, "mfpt:", mfpt_biased)
            kemeny_constant_check(M_biased.shape[0], mfpts_biased, peq)
            #we plot the F.
            
        if best_mfpt > mfpt_biased:
            best_mfpt = mfpt_biased
            best_params = np.concatenate((a, b, c)) #we concatenate the params into a single array. in shape (30,)

    print("best mfpt:", best_mfpt)
    # if not plot: 
    #     total_bias = np.zeros_like(qspace)
    #     for j in range(num_gaussian):
    #         total_bias += gaussian(qspace, best_params[j], best_params[j+num_gaussian], best_params[j+2*num_gaussian])
    #     working_bias = total_bias[working_indices]
    #     x = qspace[working_indices]
    #     plt.plot(qspace[working_indices], working_bias, label="working bias")
    #     unb_bins, unb_profile = np.load("Unbiased_Profile.npy")
    #     #we take first quarter.
    #     unb_bins = unb_bins[:len(unb_bins)//4]
    #     unb_profile = unb_profile[:len(unb_profile)//4]
    #     plt.plot(unb_bins, unb_profile, label="unbiased F")
    #     #plot the total_bias
    #     plt.plot(qspace, total_bias, label="total bias", alpha=0.3)
    #     plt.xlim(2.0, 9)
    #     plt.legend()
    #     plt.savefig(f"./bias_plots_fes_best_param_prop{i}.png")
    #     plt.close()
    #now we use the best params to local optimise the gaussian params

    def mfpt_helper(params, M, start_state = start_index, end_state = end_index, kT=0.5981, working_indices=working_indices):
        a = params[:num_gaussian]
        b = params[num_gaussian:2*num_gaussian]
        c = params[2*num_gaussian:]
        total_bias = np.zeros_like(x)
        for j in range(num_gaussian):
            total_bias += gaussian(x, a[j], b[j], c[j])

        #now we need to convert the total_bias to the working_indices space.
        #working_bias = total_bias[working_indices]
        
        #M_biased = bias_M_1D(M, total_bias, kT=0.5981)
        N = M.shape[0]
        data = []
        indices = []
        indptr = [0]

        # Iterate over non-zero elements of M to fill data, indices, and indptr
        for i in range(N):
            for j_indptr in range(M.indptr[i], M.indptr[i + 1]):
                j = M.indices[j_indptr]
                u_ij = total_bias[j] - total_bias[i]
                new_value = M.data[j_indptr] * np.exp(-u_ij / (2 * 0.5981))
                data.append(new_value)
                indices.append(j)
            indptr.append(len(data))

        # Create a new CSR matrix M_biased from data, indices, and indptr
        M_biased = config.sparse_mat[1]((data, indices, indptr), shape=(N, N))
        #print(M_biased)
        #peq,F,_,_,_,_  = compute_free_energy(M_biased.T.astype(np.float64), kT=0.5981)
        peq, F = compute_free_energy_power_method_sparse(M_biased, kT=0.5981)
        #return peq, M_biased
        mfpts_biased = Markov_mfpt_calc_sparse(peq, M_biased)
        mfpt_biased = mfpts_biased[start_index, end_index]
        return mfpt_biased

    res = minimize(mfpt_helper, #minimize comes from scipy.
                   best_params, #gaussian params
                   args=(M,
                         start_state_working_index, 
                         end_state_working_index,
                         working_indices), 
                   method='Nelder-Mead', 
                   bounds= [(0.1, 0.7)]*config.num_gaussian + [(0,2*np.pi)]*config.num_gaussian + [(0.3, 2)]*config.num_gaussian, #add bounds to the parameters
                   tol=1e0)
    return res.x



def try_and_optim_M(M, working_indices, num_gaussian=10, start_index=0, end_index=0, plot = False):
    #print("inside try and optim_M")
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
    x = np.linspace(0, 2*np.pi, config.num_bins+1) #hard coded for now.
    best_mfpt = 1e20 #initialise the best mfpt np.inf

    #first we convert the big index into "index to the working indices".

    #start_state_working_index = np.where(working_indices == start_state)[0][0] #convert start_state to the offset index space.
    #end_state_working_index = np.where(working_indices == end_state)[0][0] #convert end_state to the offset index space.
    start_state_working_index = np.argmin(np.abs(working_indices - start_index))
    end_state_working_index = np.argmin(np.abs(working_indices - end_index))
    print("optimizing to get g_param from start state:", start_state_working_index, "to end state:", end_state_working_index, "in working indices.")
    print("converted to xspace that's from:", x[working_indices[start_state_working_index]], "to", x[working_indices[end_state_working_index]])
    #now our M/working_indices could be incontinues. #N = M.shape[0]
    
    #we get the upper/lower bound of the gaussian params.
    upper = x[working_indices[-1]]
    lower = x[working_indices[0]]
    print("upper bound:", upper, "lower bound:", lower)

    for try_num in range(1000): 
        rng = np.random.default_rng()
        #we set a to be 1
        a = np.ones(num_gaussian) *0.35
        b = rng.uniform(0, 2*np.pi, num_gaussian)
        #b = rng.uniform(lower, upper, num_gaussian)
        c = rng.uniform(0.3, 2, num_gaussian)
        
        #we convert the working_indices to the qspace.

        total_bias = np.zeros_like(x)
        for j in range(num_gaussian):
            total_bias += gaussian(x, a[j], b[j], c[j])

        #now we need to convert the total_bias to the working_indices space.
        working_bias = total_bias[working_indices]
        
        #M_biased = bias_M_1D(M, total_bias, kT=0.5981)
        M_biased = np.zeros_like(M)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                u_ij = working_bias[j] - working_bias[i]
                M_biased[i, j] = M[i, j] * np.exp(-u_ij / 2*0.5981)
            M_biased[i, i] = M[i,i]
        
        for i in range(M_biased.shape[0]):
            row_sum = np.sum(M_biased[i, :])
            if row_sum > 0:
                M_biased[i, :] = M_biased[i, :] / row_sum
            else:
                M_biased[i, :] = 0


        #peq,F,_,_,_,_  = compute_free_energy(M_biased.T.astype(np.float64), kT=0.5981)
        peq, F = compute_free_energy_power_method(M_biased, kT=0.5981)
        
        mfpts_biased = Markov_mfpt_calc(peq, M_biased)
        mfpt_biased = mfpts_biased[start_state_working_index, end_state_working_index]
        #print(peq)
        #kemeny_constant_check(M.shape[0], mfpts_biased, peq)
        if try_num % 100 == 0:
            print("random try:", try_num, "mfpt:", mfpt_biased)
            kemeny_constant_check(M.shape[0], mfpts_biased, peq)
            #we plot the F.
            
        if best_mfpt > mfpt_biased:
            best_mfpt = mfpt_biased
            best_params = np.concatenate((a, b, c)) #we concatenate the params into a single array. in shape (30,)
            

    print("best mfpt:", best_mfpt)
    # if not plot: 
    #     total_bias = np.zeros_like(qspace)
    #     for j in range(num_gaussian):
    #         total_bias += gaussian(qspace, best_params[j], best_params[j+num_gaussian], best_params[j+2*num_gaussian])
    #     working_bias = total_bias[working_indices]
    #     x = qspace[working_indices]
    #     plt.plot(qspace[working_indices], working_bias, label="working bias")
    #     unb_bins, unb_profile = np.load("Unbiased_Profile.npy")
    #     #we take first quarter.
    #     unb_bins = unb_bins[:len(unb_bins)//4]
    #     unb_profile = unb_profile[:len(unb_profile)//4]
    #     plt.plot(unb_bins, unb_profile, label="unbiased F")
    #     #plot the total_bias
    #     plt.plot(qspace, total_bias, label="total bias", alpha=0.3)
    #     plt.xlim(2.0, 9)
    #     plt.legend()
    #     plt.savefig(f"./bias_plots_fes_best_param_prop{i}.png")
    #     plt.close()
    
    #now we use the best params to local optimise the gaussian params

    def mfpt_helper(params, M, start_state = start_index, end_state = end_index, kT=0.5981, working_indices=working_indices):
        a = params[:num_gaussian]
        b = params[num_gaussian:2*num_gaussian]
        c = params[2*num_gaussian:]
        total_bias = np.zeros_like(x)
        for j in range(num_gaussian):
            total_bias += gaussian(x, a[j], b[j], c[j])

        #now we need to convert the total_bias to the working_indices space.
        working_bias = total_bias[working_indices]

        #M_biased = bias_M_1D(M, total_bias, kT=0.5981)
        M_biased = np.zeros_like(M)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                u_ij = working_bias[j] - working_bias[i]
                M_biased[i, j] = M[i, j] * np.exp(-u_ij / 2*0.5981)
            M_biased[i, i] = M[i,i]
        
        for i in range(M_biased.shape[0]):
            row_sum = np.sum(M_biased[i, :])
            if row_sum > 0:
                M_biased[i, :] = M_biased[i, :] / row_sum
            else:
                M_biased[i, :] = 0
        #peq,F,_,_,_,_ = compute_free_energy(M_biased.T.astype(np.float64), kT=0.5981)
        peq, F = compute_free_energy_power_method(M_biased, kT=0.5981)
        mfpts_biased = Markov_mfpt_calc(peq, M_biased)
        mfpt_biased = mfpts_biased[start_state_working_index, end_state_working_index]

        return mfpt_biased

    res = minimize(mfpt_helper, #minimize comes from scipy.
                   best_params, #gaussian params
                   args=(M,
                         start_state_working_index, 
                         end_state_working_index,
                         working_indices), 
                   method='Nelder-Mead', 
                   bounds= [(0.1, 0.7)]*config.num_gaussian + [(0,2*np.pi)]*config.num_gaussian + [(0.3, 2)]*config.num_gaussian, #add bounds to the parameters
                   tol=1e0)
    return res.x    #, best_params
##### used
def apply_fes(system, particle_idx, gaussian_param=None, pbc = False, name = "FES", amp = 7, mode = "gaussian", plot = False, plot_path = "./fes_visualization.png"):
    """
    this function apply the bias given by the gaussian_param to the system.
    """
    pi = np.pi #we need convert this into nm.
    #unpack gaussian parameters
    if mode == "gaussian":
        num_gaussians = int(len(gaussian_param)/5)
        A = gaussian_param[0::5] * amp #*7
        x0 = gaussian_param[1::5]
        y0 = gaussian_param[2::5]
        sigma_x = gaussian_param[3::5]
        sigma_y = gaussian_param[4::5]

        #now we add the force for all gaussians.
        energy = "0"
        force = openmm.CustomExternalForce(energy)
        for i in range(num_gaussians):
            if pbc:
                energy = f"A{i}*exp(-periodicdistance(x,0,0, x0{i},0,0)^2/(2*sigma_x{i}^2) - periodicdistance(0,y,0, 0,y0{i},0)^2/(2*sigma_y{i}^2))"
                force = openmm.CustomExternalForce(energy)
            else:
                energy = f"A{i}*exp(-(x-x0{i})^2/(2*sigma_x{i}^2) - (y-y0{i})^2/(2*sigma_y{i}^2))"
                force = openmm.CustomExternalForce(energy)

            #examine the current energy term within force.

            print(force.getEnergyFunction())

            force.addGlobalParameter(f"A{i}", A[i])
            force.addGlobalParameter(f"x0{i}", x0[i])
            force.addGlobalParameter(f"y0{i}", y0[i])
            force.addGlobalParameter(f"sigma_x{i}", sigma_x[i])
            force.addGlobalParameter(f"sigma_y{i}", sigma_y[i])
            force.addParticle(particle_idx)
            #we append the force to the system.
            system.addForce(force)
        if plot:
            #plot the fes.
            x = np.linspace(0, 2*np.pi, 100)
            y = np.linspace(0, 2*np.pi, 100)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)
            for i in range(num_gaussians):
                Z += A[i] * np.exp(-(X-x0[i])**2/(2*sigma_x[i]**2) - (Y-y0[i])**2/(2*sigma_y[i]**2))
            plt.figure()
            plt.imshow(Z, cmap="coolwarm", extent=[0, 2*np.pi,0, 2*np.pi], vmin=0, vmax=amp *12/7, origin="lower")
            plt.xlabel("x")
            plt.xlim([-1, 2*np.pi+1])
            plt.ylim([-1, 2*np.pi+1])
            plt.ylabel("y")
            plt.title("FES mode = gaussian, pbc=False")
            plt.colorbar()
            plt.savefig(plot_path)
            plt.close()
            fes = Z

    if mode == "multiwell":
        """
        here we create a multiple well potential.
         essentially we deduct multiple gaussians from a flat surface, 
         with a positive gaussian acting as an additional barrier.
         note we have to implement this into openmm CustomExternalForce.
            the x,y is [0, 2pi]
         eq:
            U(x,y) = amp * (1                                                                   #flat surface
                            - A_i*exp(-(x-x0i)^2/(2*sigma_xi^2) - (y-y0i)^2/(2*sigma_yi^2))) ...        #deduct gaussians
                            + A_j * exp(-(x-x0j)^2/(2*sigma_xj^2) - (y-y0j)^2/(2*sigma_yj^2))       #add a sharp positive gaussian
        """
        if pbc:
            raise NotImplementedError("pbc not implemented for multi-well potential.")
        else:
            num_hills = 9

            #here's the well params
            A_i = np.array([0.9, 0.3, 0.7, 1, 0.2, 0.4, 0.9, 0.9, 0.9]) * amp #this is in kcal/mol.
            x0_i = [1.12, 1, 3, 4.15, 4, 5.27, 4.75, 6, 1] # this is in nm.
            sigma_x_i = [0.5, 0.3, 0.4, 2, 0.9, 1, 0.3, 0.5, 0.5]

            #now we add the force for all gaussians.
            #note all energy is in Kj/mol unit.
            energy = str(amp * 4.184) #flat surface
            force = openmm.CustomExternalForce(energy)
            force.addParticle(particle_idx)
            system.addForce(force)
            for i in range(num_hills):
                energy = f"A{i}*exp(-(x-x0{i})^2/(2*sigma_x{i}^2))"
                force = openmm.CustomExternalForce(energy)

                #examine the current energy term within force.
                print(force.getEnergyFunction())
                force.addGlobalParameter(f"A{i}", A_i[i] * 4.184) #convert kcal to kj
                force.addGlobalParameter(f"x0{i}", x0_i[i])
                force.addGlobalParameter(f"sigma_x{i}", sigma_x_i[i])
                force.addParticle(particle_idx)
                #we append the force to the system.
                system.addForce(force)
            
            if plot:
                #plot the fes.
                x = np.linspace(0, 2*np.pi, config.num_bins)
                Z = np.zeros_like(x)
                for i in range(num_hills):
                    Z += A_i[i] * 4.184 * np.exp(-(x-x0_i[i])**2/(2*sigma_x_i[i]**2))

                plt.figure()
                plt.plot(x, Z, label="multiwell FES")
                plt.xlabel("x")
                plt.xlim([0, 2*np.pi])
                plt.title("FES mode = 1D multiwell, pbc=False")
                plt.savefig(plot_path)
                plt.close()
                fes = Z
            
    if mode == "funnel":
        """
        this is funnel like potential.
        we start wtih a flat fes, then add/deduct sphrical gaussians
        eq:
            U = 0.7* amp * cos(2 * p * (sqrt((x-pi)^2 + (y-pi)^2))) #cos function. periodicity determines the num of waves.
            - amp exp(-((x-pi)^2+(y-pi)^2))
            + 0.4*amp*((x-pi/8)^2 + (y-pi/8)^2)
        """
        if pbc:
            raise NotImplementedError("pbc not implemented for funnel potential.")
        else:
            plot_3d = False
            periodicity = 8
            energy = f"0.7*{amp} * cos({periodicity} * (sqrt((x-{pi})^2 + (y-{pi})^2))) - 0.6* {amp} * exp(-((x-{pi})^2+(y-{pi})^2)) + 0.4*{amp}*((x-{pi}/8)^2 + (y-{pi}/8)^2)"
            
            force = openmm.CustomExternalForce(energy)
            force.addParticle(particle_idx)
            system.addForce(force)
            if plot:
                if plot_3d:
                    import plotly.graph_objs as go

                    # Define the x, y, and z coordinates
                    x = np.linspace(0, 2*np.pi, 100)
                    y = np.linspace(0, 2*np.pi, 100)
                    X, Y = np.meshgrid(x, y)
                    Z = np.zeros_like(X)
                    Z += 0.9* amp * np.cos(periodicity * (np.sqrt((X-np.pi)**2 + (Y-np.pi)**2))) #cos function. periodicity determines the num of waves.
                    Z -= 0.6* amp * np.exp(-((X-np.pi)**2/0.5+(Y-np.pi)**2)/0.5)
                    Z += 0.4*amp*(((X-np.pi)/8)**2 + ((Y-np.pi)/8)**2)

                    # Create the 3D contour plot
                    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, cmin = 0, cmax = amp *12/7)])
                    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                  highlightcolor="limegreen", project_z=True))
                    fig.update_layout(title='FES mode = funnel, pbc=False', autosize=True,
                                    width=800, height=800,
                                    scene = {
                                        "xaxis": {"nticks": 5},
                                        "yaxis": {"nticks": 5},
                                        "zaxis": {"nticks": 5},
                                        "camera_eye": {"x": 1, "y": 1, "z": 0.4},
                                        "aspectratio": {"x": 1, "y": 1, "z": 0.4}
                                    }
                                    )
                                    #margin=dict(l=65, r=50, b=65, t=90))
                    #save fig.
                    fig.write_image(plot_path)
                    fes = Z
                    
                else:
                    #plot the fes.
                    x = np.linspace(0, 2*np.pi, 100)
                    y = np.linspace(0, 2*np.pi, 100)
                    X, Y = np.meshgrid(x, y)
                    Z = np.zeros_like(X)
                    Z += 0.4* amp * np.cos(periodicity * (np.sqrt((X-np.pi)**2 + (Y-np.pi)**2))) #cos function. periodicity determines the num of waves.
                    Z += 0.7* amp * np.exp(-((X-np.pi)**2/0.5+(Y-np.pi)**2/0.5))
                    Z += 0.2*amp*(((X-np.pi)/8)**2 + ((Y-np.pi)/8)**2)

                    plt.figure()
                    plt.imshow(Z, cmap="coolwarm", extent=[0, 2*np.pi,0, 2*np.pi], vmin=0, vmax=amp *12/7, origin="lower")
                    plt.xlabel("x")
                    plt.xlim([-1, 2*np.pi+1])
                    plt.ylim([-1, 2*np.pi+1])
                    plt.ylabel("y")
                    plt.title("FES mode = funnel, pbc=False")
                    plt.colorbar()
                    plt.savefig(plot_path)
                    plt.close()
                    fes = Z

    #at last we add huge barrier at the edge of the box. since we are not using pbc.
    #this is to prevent the particle from escaping the box.
    # if x<0, push the atom back to x=0
    left_pot = openmm.CustomExternalForce("1e10 * step(-x)")
    right_pot = openmm.CustomExternalForce(f"1e10 * step(-(x - 2*{pi}))")
    #bottom_pot = openmm.CustomExternalForce("1e10 * step(-y)")
    #top_pot = openmm.CustomExternalForce(f"1e10 * step(-(y - 2*{pi}))")

    left_pot.addParticle(particle_idx)
    right_pot.addParticle(particle_idx)
    #bottom_pot.addParticle(particle_idx)
    #top_pot.addParticle(particle_idx)

    system.addForce(left_pot)
    system.addForce(right_pot)
    #system.addForce(bottom_pot)
    #system.addForce(top_pot)

    return system, fes #return the system and the fes (2D array for plotting.)
### used
def apply_bias(system, particle_idx, gaussian_param, pbc = False, name = "BIAS", num_gaussians = 20):
    """
    this applies a bias using customexternal force class. similar as apply_fes.
    note this leaves a set of global parameters Ag, x0g, sigma_xg
    as these parameters can be called and updated later.
    note this is done while preparing the system before assembling the context.
    """
    assert len(gaussian_param) == 3 * num_gaussians, "gaussian_param should be in A, x0, sigma_x, format."

    #unpack gaussian parameters gaussian_params = np.concatenate((a, bx, cx))
    num_gaussians = len(gaussian_param)//3
    A = gaussian_param[:num_gaussians]
    x0 = gaussian_param[num_gaussians:2*num_gaussians]
    sigma_x = gaussian_param[2*num_gaussians:3*num_gaussians]

    #now we add the force for all gaussians. with num_gaussians terms.
    energy = "0"
    force = openmm.CustomExternalForce(energy)
    for i in range(num_gaussians):
        if pbc:
            raise NotImplementedError("pbc not implemented for gaussian potential.")
            energy = f"Ag{i}*exp(-periodicdistance(x,0,0, x0g{i},0,0)^2/(2*sigma_xg{i}^2) - periodicdistance(0,y,0, 0,y0g{i},0)^2/(2*sigma_yg{i}^2))"
            force = openmm.CustomExternalForce(energy)
        else:
            energy = f"Ag{i}*exp(-(x-x0g{i})^2/(2*sigma_xg{i}^2))" #in openmm unit, kj/mol, nm.
            force = openmm.CustomExternalForce(energy)

        #examine the current energy term within force.

        print(force.getEnergyFunction())

        force.addGlobalParameter(f"Ag{i}", A[i] * 4.184) #convert to kJ/mol
        force.addGlobalParameter(f"x0g{i}", x0[i]) #convert to nm
        force.addGlobalParameter(f"sigma_xg{i}", sigma_x[i])
        force.addParticle(particle_idx)
        #we append the force to the system.
        system.addForce(force)
    
    print("system added with bias.")
    return system
##### used
def update_bias(simulation, gaussian_param, name = "BIAS", num_gaussians = 20):
    """
    given the gaussian_param, update the bias
    note this requires the context object. or a simulation object.
    # the context object can be accessed by simulation.context.
    """
    assert len(gaussian_param) == 3 * num_gaussians, "gaussian_param should be in A, x0, sigma_x, format."

    #unpack gaussian parameters gaussian_params = np.concatenate((a, bx, cx))
    num_gaussians = len(gaussian_param)//3
    A = gaussian_param[:num_gaussians]
    x0 = gaussian_param[num_gaussians:2*num_gaussians]
    sigma_x = gaussian_param[2*num_gaussians:3*num_gaussians]

    #now we update the GlobalParameter for all gaussians. with num_gaussians terms. and update them in the system.
    #note globalparameter does NOT need to be updated in the context.
    for i in range(num_gaussians):
        simulation.context.setParameter(f"Ag{i}", A[i] * 4.184) #convert to kJ/mol
        simulation.context.setParameter(f"x0g{i}", x0[i]) #convert to nm
        simulation.context.setParameter(f"sigma_xg{i}", sigma_x[i])
    
    print("system bias updated")
    return simulation
##### used
def get_total_bias(x, gaussian_param, num_gaussians = 20):
    """
    this function returns the total bias given the gaussian_param.
    note this is used for plotting the total bias.
    """
    assert len(gaussian_param) == 3 * num_gaussians, "gaussian_param should be in A, x0, sigma_x, format."

    #unpack gaussian parameters gaussian_params = np.concatenate((a, bx, cx))
    num_gaussians = len(gaussian_param)//3
    A = gaussian_param[:num_gaussians]
    x0 = gaussian_param[num_gaussians:2*num_gaussians]
    sigma_x = gaussian_param[2*num_gaussians:3*num_gaussians]

    #now we add the force for all gaussians. with num_gaussians terms.
    total_bias = np.zeros_like(x)
    for i in range(num_gaussians):
        total_bias += A[i] * np.exp(-(x-x0[i])**2/(2*sigma_x[i]**2))
    
    return total_bias
