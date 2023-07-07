#Written by TW 16th May
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
from torch.nn import functional as F

# this is utility function for main.py

#define a function calculating free energy
#original matlab code:  [pi, F, eigenvectors, eigenvalues, eigenvalues_sorted, index]=compute_free_energy(K, kT)

def gaussian(x, a, b, c): #self-defined gaussian function
    return a * np.exp(-(x - b)**2 / 2*(c**2)) 
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

    K = np.zeros((N,N))
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
    onevec = np.ones((N, 1))
    Qinv = np.linalg.inv(peq.T * onevec - K.T)

    mfpt = np.zeros((N, N))
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
    K_biased = np.zeros([N, N])

    for i in range(N-1):
        u_ij = total_bias[i+1] - total_bias[i]  

        K_biased[i, i+1] = K[i, i+1] * np.exp(u_ij /(2*kT))  
        K_biased[i+1, i] = K[i+1, i] * np.exp(-u_ij /(2*kT))
    
    for i in range(N):
        K_biased[i,i] = -np.sum(K_biased[:,i])
    return K_biased

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
    
    #result = kemeny_constant_check(N, mfpt, peq)
    return mfpt

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
    #take the real, positive part of the eigenvector i.e. the probability distribution at equilibrium.
    peq = np.real(peq) #take the real part of the eigenvector
    peq = np.maximum(peq, 0) #take the positive part of the eigenvector

    #calculate the free energy
    F = -kT * np.log(peq + 1e-6) #add a small number to avoid log(0))

    return [peq, F, evectors, evalues, evalues_sorted, index]


#for the mingpt:
@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None, actions=None, rtgs=None, timesteps=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        # x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        x_cond = x if x.size(1) <= block_size//3 else x[:, -block_size//3:] # crop context if needed
        if actions is not None:
            actions = actions if actions.size(1) <= block_size//3 else actions[:, -block_size//3:] # crop context if needed
        rtgs = rtgs if rtgs.size(1) <= block_size//3 else rtgs[:, -block_size//3:] # crop context if needed
        logits, _ = model(x_cond, actions=actions, targets=None, rtgs=rtgs, timesteps=timesteps)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        # x = torch.cat((x, ix), dim=1)
        x = ix

    return x