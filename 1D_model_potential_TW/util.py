#Written by TW 16th May
import numpy as np
from scipy.linalg import eig
from scipy.linalg import expm
# this is utility function for main.py

#define a function calculating free energy
#original matlab code:  [pi, F, eigenvectors, eigenvalues, eigenvalues_sorted, index]=compute_free_energy(K, kT)

def gaussian(x, a, b, c): #self-defined gaussian function
        return a * np.exp(-(x - b)**2 / ((2*c)**2)) 

def create_K_1D(N, kT):
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
    
    K = np.zeros((N,N))
    for i in range(N-1):
        K[i, i + 1] = amplitude * np.exp((y[i+1] - y[i]) / 2 / kT)
        K[i + 1, i] = amplitude * np.exp((y[i] - y[i+1]) / 2 / kT) #where does this formula come from?
    for i in range(N):
        K[i, i] = 0
        K[i, i] = -np.sum(K[:, i])
    return K

def compute_free_energy(K, kT):
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
    F = -kT * np.log(peq + 1e-9) #add a small number to avoid log(0))

    return [peq, F, evectors, evalues, evalues_sorted, index]

def kemeny_constant_check(N, mfpt, peq):
    kemeny = np.zeros((N, 1))
    for i in range(N):
        for j in range(N):
            kemeny[i] = kemeny[i] + mfpt[i, j] * peq[j]
    #print("Performing Kemeny constant check...")
    #print("the min/max of the Kemeny constant is:", np.min(kemeny), np.max(kemeny))
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
    onevec = np.ones((N, 1))
    Qinv = np.linalg.inv(peq.T @ onevec - K.T) #Qinv is the inverse of the matrix Q 

    mfpt = np.zeros((N, N))
    for j in range(N):
        for i in range(N):
            #to avoid devided by zero error:
            if peq[j] == 0:
                mfpt[i, j] = 0
            else:
                mfpt[i, j] = 1 / peq[j] * (Qinv[j, j] - Qinv[i, j])
    
    result = kemeny_constant_check(N, mfpt, peq)
    return mfpt

#here we define a function, transform the unperturbed K matrix,
#with given biasing potential, into a perturbed K matrix K_biased.

def bias_K_1D(K, total_bias, kT, N, cutoff = 20):
    """
    K is the unperturbed transition matrix.
    total_bias is the total biasing potential.
    kT is the thermal energy.
    N is the number of states.

    This function returns the perturbed transition matrix K_biased.
    """
    K_biased = np.zeros([N, N])
    for i in range(N-1):
        u_ij = total_bias[i+1] - total_bias[i]  # Calculate u_ij (Note: Indexing starts from 0)
        if u_ij > cutoff: u_ij = cutoff  # Apply cutoff to u_ij
        if u_ij < -cutoff: u_ij = -cutoff

        K_biased[i, i+1] = K[i, i+1] * np.exp(-u_ij /(2*kT))  # Calculate K_biased
        K_biased[i+1, i] = K[i+1, i] * np.exp(u_ij /(2*kT))
        K_biased[i,i] = 0 #kinda redundant.
    
    for i in range(N):
        K_biased[i,i] = -np.sum(K_biased[:,i])
    return K_biased.T

#here is the python equivalent of the matlab function "explore_the_network.m"

def explore_the_network(t_max, ts, K_biased, state_start, state_end,N=100):
    """
    Random walk style simulation of the biased network.
    t_max is the maximum simulation time.
    ts is the time step.
    K_biased is the biased transition matrix.
    kT is the thermal energy.
    cutoff is the cutoff for the biasing potential.
    nodes is ?
    state_start is the starting state.
    state_end is the ending state.
    """

    #M_t is the transition matrix, calculated by yielding the exponential of sq matrix K_biiased times the time step ts, then transpose.
    #this gives us the population transition rate of the i,j element per time step.
    M_t =expm(K_biased * ts)  #the rate is K_biased * ts. The transition matrix is the exponential of the rate matrix.
    
    #normalize the M_t matrix, such that each row sums to 1.
    M_t = M_t / np.sum(M_t, axis=1, keepdims=True) #do we need normalize this?

    record_states = np.zeros(t_max)
    record_states[0] = state_start
    
    # Perform the Simulation
    for i in range(1, t_max):
        P = M_t[int(record_states[i-1]), :] #P is the probability distribution of the next state
        
        new_state = np.random.choice(N, 1, p=P) #we use np.random.choice to decide the next state
        
        record_states[i] = new_state
        
        if new_state == state_end:
            print("Reached the end state at time step %d" % (i))
            step = i #steps needed to reach the end state.
            break
    return record_states, step


def markov_mfpt_calc(peq, M):
    """
    peq is the equilibrium probability distribution.
    M is the transition matrix.
    """
    N = M.shape[0]
    onevec = np.ones(N)
    I = np.diag(onevec)
    A = np.outer(peq, onevec)
    Qinv = np.linalg.inv(A + I - M)
    mfpt = np.zeros((N, N))
    
    for j in range(N):
        for i in range(N):
            mfpt[i, j] = 1 / peq[j] * (Qinv[j, j] - Qinv[i, j] + I[i, j])
    
    result = kemeny_constant_check(N, mfpt, peq)
    return mfpt


def min_mfpt(abc_init, K, num_gaussian, kT, ts, state_start, state_end):
    """
    abc_init is the parameters for gaussians, namely the a, b c in the gaussian function.
     inside abc_init: [a, b, c]
    K is the unperturbed transition matrix.
    """
    N = K.shape[0]

    a, b, c = abc_init[0], abc_init[1], abc_init[2]

    gaussian_functions = []
    for i in range(num_gaussian):
        gaussian_functions.append(gaussian(np.linspace(0, 99, 100), a[i], b[i], c[i])) #5 gaussians start from 0 to 99.

    #we sum up all the random gaussian functions as total_bias
    total_bias = np.sum(gaussian_functions, axis=0)

    K_biased = bias_K_1D(K, total_bias, kT, N=100, cutoff=20)
    peq_biased = compute_free_energy(K_biased, kT)[0]
    
    M_t = expm(K_biased*ts)
    Mmfpt_biased = markov_mfpt_calc(peq_biased.T, M_t) * ts  # we use markov probagation as updating rule.
    return Mmfpt_biased[state_start, state_end]



"""
To be Done in the future.
#python equivalent of DHAM_unbias.m
def DHAM_unbias(record_states, x_eq, kT, N, bias, force_constant=0.1, cutoff=20):
    qspace = np.linspace(0.9, N + 1, N + 1)
    num_sims = record_states.shape[0]
    dat_lengths = record_states.shape[1]
    lagtime = 1

    ncount = np.zeros((num_sims, N))

    for k in range(num_sims):
        ncount[k, :] = np.histogram(record_states[k, :dat_lengths[k] - 1], bins=qspace)[0]

    MM = np.zeros((N, N))

    for k in range(num_sims):
        b = np.zeros(dat_lengths[k])
        for i in range(dat_lengths[k]):
            hist = np.histogram(record_states[k, i], bins=qspace)
            b[i] = np.argmax(hist[0])
        
        for i in range(lagtime, dat_lengths[k]):
            msum = 0
            for l in range(num_sims):
                nc = ncount[l, int(b[i - lagtime])]
                epot_j = 0.5 * force_constant * (bias[l, int(b[i])] - x_eq[l]) ** 2
                epot_i = 0.5 * force_constant * (bias[l, int(b[i - lagtime])] - x_eq[l]) ** 2
                delta_epot = max(-cutoff, epot_j - epot_i)
                delta_epot = min(cutoff, delta_epot)
                if nc > 2:
                    msum += nc * np.exp(-delta_epot / kT / 2)
            
            if msum > 0:
                MM[int(b[i - lagtime]), int(b[i])] += 1 / msum

    msum = MM.sum(axis=1)
    for i in range(N):
        if msum[i] > 0:
            MM[i, :] /= msum[i]
        else:
            MM[i, :] = 0

    eigvals, eigvecs = eig(MM.T)
    l = np.argsort(np.real(eigvals))
    prob_dist = eigvecs[:, l[N-1]].real
    prob_dist /= np.sum(prob_dist)
    relax_time = -lagtime / np.log(eigvals[l[N-2]].real)

    return prob_dist

"""
