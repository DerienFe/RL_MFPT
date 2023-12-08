from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from os import path
plt.rcParams.update({'font.size': 20})

#here we load the precalculated 1D optim bias and zero it on 89 state.
# note all loaded data is gaussian parameters in 1D.
#state_start = 7 #note in this code this is 0-indexed. so 8 means state 9.
state_end = 88
N = 100 #number of grid points, i.e. num of states.
kT = 0.5981
#qspace is from 0 to 99, with 100 states.
qspace = np.linspace(0, 99, 100)
cur_dir = path.dirname(path.realpath(__file__))
filename = path.join(cur_dir, 'pos_bias_m.mat')
gaussian_params_pos = loadmat(filename)['pos_bias'][0]


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
    y = xtilt*y1 - y2
    y = (xtilt*y1 + (1-xtilt)*y2)
    y = y*3
    
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

def gaussian_1d(x, bx, cx):
    return np.exp(-((x-bx)**2/(2*cx**2)))

def get_total_bias_1d(x, c_g, std_g):
    total_bias = np.zeros(x.shape[0])
    for i in range(20):
        total_bias += gaussian_1d(x, c_g[i], std_g[i])
    return total_bias


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


#test plot the K fes.
K = create_K_1D(N, kT)
F = compute_free_energy(K, kT)[1]
F -= F[88]

#we truncate the F until 88.
F = F[:89]

#plt.plot(F)
#plt.show()
colormap = plt.cm.get_cmap('coolwarm', gaussian_params_pos.shape[0])
#plot the FES. up until 88.
plt.figure(figsize=(8,6))
plt.plot(88, F[88], marker = 'x', color = 'red', markersize = 10)
#plt.plot(pos_i, F[pos_i], marker = 'o', color = 'red', markersize = 10)
plt.plot(F, color = 'black', linewidth = 2)
F_biased_total=[]
for pos_i in range(0, gaussian_params_pos.shape[0],3):
    #unpack all the gaussian params. 20 center_gaussian and 20 std gaussian.
    allparam = gaussian_params_pos[pos_i][0]

    c_g = allparam[:20]
    std_g = allparam[20:]

    #get the total gaussian bias.
    total_bias = get_total_bias_1d(qspace, c_g, std_g)

    #now we apply this bias on K and calculate FES.
    K_biased = bias_K_1D(K, total_bias, kT)
    F_biased = compute_free_energy(K_biased, kT)[1]

    #truncate the FES until 88.
    F_biased = F_biased[:89]
    #zero the F_biased on state 89.
    F_biased -= F_biased[88]
    F_biased_total.append(F_biased)

    
    #plot the current position on F.
    
    for i in range(len(F_biased_total)):
        plt.plot(F_biased_total[i], color = colormap(i*3), alpha = 0.7)

    #plot agin the current position on F.
    #plt.plot(pos_i, F[pos_i], marker = 'o', color = 'red', markersize = 10)
    #plot the current position on lastest F_biased.
    #plt.plot(pos_i, F_biased_total[-1][pos_i], marker = 'o', color = 'red', markersize = 10, alpha = 0.5)
    
#position legend on the top right with alpha=0.7
plt.legend()
plt.xlabel('state')
plt.ylabel('FES (kcal/mol)')
plt.tight_layout()
#plt.show()
plt.savefig(f'./figs/1D_gif/optim-{pos_i}_overlap.png')
plt.close()



import imageio
import os

path = './figs/1D_gif/'

# Get all file names sorted by their creation time
file_names = sorted(os.listdir(path), key=lambda x: int(x.split('-')[1].split('_')[0]) if '_overlap' in x and x.endswith('.png') else 999999)

# Create a writer object
writer = imageio.get_writer('./figs/1D_gif/animation_2.gif', fps=3)

# Add images to the writer object
for file_name in file_names:
    if 'overlap' in file_name and file_name.endswith('.png'):
        writer.append_data(imageio.imread(os.path.join(path, file_name)))

# Close the writer object
writer.close()

print('done')
