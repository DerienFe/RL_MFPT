#reduced util file for DHAM RL env

#Written by TW 16th May
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import logm, expm
import random
# this is utility function for main.py

#define a function calculating free energy
#original matlab code:  [pi, F, eigenvectors, eigenvalues, eigenvalues_sorted, index]=compute_free_energy(K, kT)

def gaussian(x, a, b, c): #self-defined gaussian function
    """
    x is the x-axis, np array.
    """
    return a * np.exp(-(x - b)**2 / 2 / c**2)
    
def real_nonnegative(M): #filter to make sure the M matrix is all positive and real.
    M = M.real
    M = np.where(M < 0, 0, M)
    return M

def create_K_1D(N=100, kT=0.5981):
    #create the K matrix for 1D model potential
    #K is a N*N matrix, representing the transition rate between states
    #The diagonal elements are the summation of the other elements in the same row, i.e. the overall outflow rate from state i
    #The off-diagonal elements are the transition rate from state i to state j (or from j to i???)
    x = np.linspace(0, 5*np.pi, N) #create a grid of x values
    y1 = np.sin((x-np.pi))
    y2 = np.sin((x-np.pi)/2)
    amplitude = 10
    xtilt = 0.5
    y = (xtilt*y1 + (1-xtilt)*y2) * 3 
    
    #here we plot the xy
    #plt.plot(x, y)
    #plt.show()

    K = np.zeros((N,N)) #, dtype=np.float64
    for i in range(N-1):
        K[i, i + 1] = amplitude * np.exp((y[i+1] - y[i]) / 2 / kT)
        K[i + 1, i] = amplitude * np.exp((y[i] - y[i+1]) / 2 / kT) #where does this formula come from?
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
#here we define a function, transform the unperturbed K matrix,
#with given biasing potential, into a perturbed K matrix K_biased.

def bias_K_1D(K, total_bias, kT=0.596):
    """
    K is the unperturbed transition matrix.
    total_bias is the total biasing potential.
    kT is the thermal energy.
    This function returns the perturbed transition matrix K_biased.
    """
    N = K.shape[0]
    K_biased = np.zeros([N, N])#, #dtype=np.float64)

    for i in range(N-1):
        u_ij = total_bias[i+1] - total_bias[i]  

        K_biased[i, i+1] = K[i, i+1] * np.exp(u_ij /(2*kT))  
        K_biased[i+1, i] = K[i+1, i] * np.exp(-u_ij /(2*kT))
    
    for i in range(N):
        K_biased[i,i] = -np.sum(K_biased[:,i])
    return K_biased

def calc_M(K, time_step, kT=0.596):
    return expm(K * time_step)

def bias_M(M, total_bias, time_step, kT = 0.596):
    #we keep the logm(M) real part.
    K = logm(M).real / time_step

    #normalize the K matrix
    for i in range(K.shape[0]):
        K[i,i] = -np.sum(K[:,i])

    K_biased = bias_K_1D(K, total_bias, kT)
    return expm(K_biased * time_step)

def simulate(state_start, state_end, M, steps = 1000):
    """
    Given the starting state, Markov Matrix (M) and the simulation steps, simulate the trajectory.
    returns:
    traj: the trajectory
    i: the number of steps taken
    cur_pos: the state at the end of traj
    Finished: whether the simulation reaches the end state


    #no need to use np.clip, because we pick next state based on the probability distribution.
    """
    M = M.T
    traj = np.zeros(steps, dtype=np.int32) #record the trajectory
    traj[0] = state_start

    cur_pos = state_start
    for i in range(steps):
        next_pos = np.random.choice(np.arange(len(M)), p=M[cur_pos])
        traj[i] = next_pos
        cur_pos = next_pos
        if i % 500 == 0:
            print("simulating at step: ", i, "cuurent position is: ", cur_pos)
        if next_pos == state_end:
            print('simulation ended at end point')
            return [traj, i+1, cur_pos, True]
    
    print("simulation ended at point:", cur_pos)
    return [traj, i+1, cur_pos, False]

def compute_free_energy(K, kT=0.596):
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
    evalues, evectors = np.linalg.eig(K)

    #sort the eigenvalues and eigenvectors
    index = np.argsort(evalues) #sort the eigenvalues, the largest eigenvalue is at the end of the list
    evalues_sorted = evalues[index] #sort the eigenvalues based on index

    #calculate the equilibrium distribution
    peq = evectors[:, index[-1]].T/np.sum(evectors[:, index[-1]]) #normalize the eigenvector
    #take the real part of the eigenvector i.e. the probability distribution at equilibrium.
    #print('sum of the peq is:', np.sum(peq))

    #calculate the free energy
    F = -kT * np.log(peq + 1e-6) #add a small number to avoid log(0))

    return [peq, F, evectors, evalues, evalues_sorted, index]

def extract_diagonal_band(matrix, band_width=7):
    """
    This function extracts a band around the diagonal of a 2D matrix.
    
    Args:
    matrix: 2D numpy array or PyTorch tensor
    band_width: the width of the band around the diagonal

    Returns:
    band: a 2D tensor of shape (N, band_width), where N is the number of states
    """
    assert band_width % 2 == 1, "band_width must be an odd number"
    offset = band_width // 2

    N = matrix.shape[0]
    band = [matrix[i, max(0, i-offset):min(N, i+offset+1)] for i in range(N)]
    band = torch.stack(band)

    # If the band width is less than the desired band width due to boundary conditions, pad with zeros
    padding = band_width - band.shape[1]
    if padding > 0:
        band = torch.nn.functional.pad(band, (0, padding))
    
    return band

# Example usage:
# M = torch.randn(100, 100)
# band = extract_diagonal_band(M, band_width=7)

def set_randomseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return None
