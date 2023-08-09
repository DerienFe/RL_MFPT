# this is util functions for 2D model.
# Written by TW 23th May

import numpy as np
from scipy.linalg import eig
from scipy.linalg import expm
from math import pi

#first we define a 2d Gaussian function
#the center is (x0, y0)
#the width is sigma_x, sigma_y
def gaussian_2d(x, y, x0, y0, sigma_x, sigma_y):
    return np.exp(-((x-x0)**2/(2*sigma_x**2) + (y-y0)**2/(2*sigma_y**2)))

#initialize a 2d potential surface. default N=100
def create_K_2D(N, kT):
    """
    N is the total state number
    kT is the thermal energy

    here we create a 2D potential surface. [100,100]
    """
    x, y = np.meshgrid(np.linspace(0, 5*np.pi, N), np.linspace(0, 5*pi, N))
    
    #create a 2D potential surface. z is the energy surface.
    tilt = 0.5
    z1 = np.sin(x-np.pi) + np.sin(y-np.pi) #the first local minima is around (pi, pi)
    z2 = np.sin(x-2.5*np.pi) + np.sin(y-3.5*np.pi) #the second local minima is around (2.5*pi, 3.5*pi)
    z = tilt * z1 + (1-tilt) * z2

    amp = 10

    K = np.zeros((N*N, N*N)) # we keep K flat. easier to do transpose etc.

    for i in range(N):
        for j in range(N):
            index = i*N + j # flatten 2D indices to 1D
            if i < N - 1: # Transition rates between vertically adjacent cells
                index_down = (i+1)*N + j # Index of cell below current cell
                delta_z = z[i+1,j] - z[i,j]
                K[index, index_down] = amp * np.exp(-delta_z / (2 * kT))
                K[index_down, index] = amp * np.exp(delta_z / (2 * kT))
            if j < N - 1: # Transition rates between horizontally adjacent cells
                index_right = i*N + j + 1 # Index of cell to the right of current cell
                delta_z = z[i,j+1] - z[i,j]
                K[index, index_right] = amp * np.exp(-delta_z / (2 * kT))
                K[index_right, index] = amp * np.exp(delta_z / (2 * kT))
    
    # Filling diagonal elements with negative sum of rest of row
    for i in range(N*N):
        K[i, i] = -np.sum(K[i, :])

    return K

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
    evalues, evectors = np.linalg.eig(K)

    #sort the eigenvalues and eigenvectors
    index = np.argsort(evalues) #sort the eigenvalues, the largest eigenvalue is at the end of the list
    evalues_sorted = evalues[index] #sort the eigenvalues based on index

    #calculate the equilibrium distribution
    peq = evectors[:, index[-1]].T/np.sum(evectors[:, index[-1]]) #normalize the eigenvector
    peq = peq.reshape((N, N)) # stationary distribution at k, l

    #take the real part of the eigenvector i.e. the probability distribution at equilibrium.
    #print('sum of the peq is:', np.sum(peq))

    #calculate the free energy
    F = -kT * np.log(peq + 1e-15) #add a small number to avoid log(0))
    F = F.reshape((N, N))
    return [peq, F, evectors, evalues, evalues_sorted, index]

def kemeny_constant_check(N, mfpt, peq):
    """
    N is the number of states in one dimension (total states is N*N)
    mfpt is the mean first passage time matrix in shape (N, N, N, N)
    peq is the stationary distribution
    """
    kemeny = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    kemeny[i, j] += mfpt[i, j, k, l] * peq[k, l]

    #print the min/max of the Kemeny constant
    print("the min/max of the Kemeny constant is:", np.min(kemeny), np.max(kemeny))

    return 

def mfpt_calc_2D(peq, K):
    """
    peq is the probability distribution at equilibrium.
    K is the transition matrix.
    N is the number of states.

    here we output the mfpt in shape (N, N, N, N).
    each element is the mfpt from (i, j) to (k, l).
    """
    N = int(np.sqrt(K.shape[0])) # Total states is N*N for a 2D grid
    onevec = np.ones((N*N, 1))
    peq_flat = peq.flatten()
    Qinv = np.linalg.inv(peq_flat.reshape(-1,1).T @ onevec - K.T) # Qinv is the inverse of the matrix Q 

    mfpt = np.zeros((N*N, N*N))
    for l in range(N):
        for k in range(N):
            for j in range(N):
                for i in range(N):
                    # convert 2D indices to 1D index
                    idx_from = i * N + j
                    idx_to = k * N + l
                    # to avoid divided by zero error:
                    if peq_flat[idx_to] == 0:
                        mfpt[idx_from, idx_to] = 0
                    else:
                        mfpt[idx_from, idx_to] = 1 / peq_flat[idx_to] * (Qinv[idx_to, idx_to] - Qinv[idx_from, idx_to])
        
    # Reshape mfpt into 2D grid shape
    # so that mfpt[i,j,k,l] is the MFPT from state (i,j) to (k,l)
    mfpt = mfpt.reshape((N, N, N, N)) 
    result = kemeny_constant_check(N, mfpt, peq)
    return mfpt

def bias_K_2D(K, total_bias, kT, cutoff = 20):
    """
    K is the rate matrix in shape (N*N, N*N)
    total_bias is the total bias potential in shape (N, N)
    kT is the thermal energy
    N is the number of states in one dimension (total states is N*N)
    """

    N = np.sqrt(K.shape[0]).astype(int)

    K_biased = np.zeros((N*N, N*N))

    for i in range(N):
        for j in range(N):
            index = i*N + j # flatten 2D indices to 1D
            if i < N - 1: 
                index_down = (i+1)*N + j
                delta_z = total_bias[i+1,j] - total_bias[i,j]
                #apply cutoff here.
                if delta_z > cutoff:
                    delta_z = cutoff
                elif delta_z < -cutoff:
                    delta_z = -cutoff
                K_biased[index, index_down] = K[index, index_down] * np.exp(-delta_z / (2 * kT))
                K_biased[index_down, index] = K[index_down, index] * np.exp(delta_z / (2 * kT))
            if j < N - 1:
                index_right = i*N + j + 1
                delta_z = total_bias[i,j+1] - total_bias[i,j]
                #apply cutoff here.
                if delta_z > cutoff:
                    delta_z = cutoff
                elif delta_z < -cutoff:
                    delta_z = -cutoff
                K_biased[index, index_right] = K[index, index_right] * np.exp(-delta_z / (2 * kT))
                K_biased[index_right, index] = K[index_right, index] * np.exp(delta_z / (2 * kT))
            
    for i in range(N*N):
        K_biased[i, i] = -np.sum(K[:, i])
    
    return K_biased

def markov_mfpt_calc_2D(peq, M):
    """
    peq is the stable distribution, in shape (N, N)
    M is the transition matrix, in shape (N*N, N*N)
    
    same as above, the output mfpt is in shape (N, N, N, N).
    each element is the mfpt from (i, j) to (k, l).
    """
    N = np.sqrt(M.shape[0]).astype(int)

    onevec = np.ones((N*N, 1))
    I = np.diag(np.ones(N*N))
    A = np.outer(peq.flatten(), onevec.flatten())

    Qinv = np.linalg.inv(A + I - M)
    mfpt = np.zeros((N*N, N*N))

    for l in range(N):
        for k in range(N):
            for j in range(N):
                for i in range(N):
                    # convert 2D indices to 1D index
                    idx_from = i * N + j
                    idx_to = k * N + l
                    # to avoid divided by zero error:
                    if peq.flatten()[idx_to] == 0:
                        mfpt[idx_from, idx_to] = 0
                    else:
                        mfpt[idx_from, idx_to] = 1 / peq.flatten()[idx_to] * (Qinv[idx_to, idx_to] - Qinv[idx_from, idx_to])
    
    mfpt = mfpt.reshape((N, N, N, N))
    result = kemeny_constant_check(N, mfpt, peq)
    return mfpt

    




