#This is a data generator for the offline RL training.
from RL_Qagent_1D.util import *
import torch

def gen_1D_20N_10Gaussian(N=20, num_gaussians=10, num_data=1000, kT=0.596, state_start=2, state_end=18):
    """
    Generate 1D data with N = 20 and 10 total gaussian bias.
    """
    data = []
    K = create_K_1D(N=N, kT=kT)
    mfpt_init = mfpt_calc(compute_free_energy(K, kT=kT)[0], K)[state_start, state_end]
    
    for _ in range(num_data):
        K = create_K_1D(N=N, kT=kT) #reset the K.
        action = [] #keep track of the action taken.

        for j in range(num_gaussians):
            position = np.random.randint(0, N)
            action.append(position)
            gaussian_bias = gaussian(np.linspace(0, N-1, N), a = 1, b = position, c = 1)

            #generate the perturbed transition matrix K_biased.
            K_biased = bias_K_1D(K, gaussian_bias, kT=kT)
        
            [peq, F, evectors, evalues, evalues_sorted, index] = compute_free_energy(K_biased, kT=kT)
            mfpts_biased = markov_mfpt_calc(peq, K_biased)
            mfpt_biased = mfpts_biased[state_start, state_end]
        
            reward = (mfpt_init - mfpt_biased) / mfpt_init

            next_state = K_biased 
            if j == num_gaussians - 1:
                done = True
            else:
                done = False

            #append the data to the data list. element in torch tensor, dtype = float64.
            data.append([torch.tensor(K, dtype=torch.float64), 
                         action, 
                         torch.tensor(reward, dtype=torch.float64), 
                         torch.tensor(next_state, dtype=torch.float64), 
                         torch.tensor(done, dtype = torch.bool)])
            K = K_biased #update the K.
    
    print(len(data))
    #save the data.
    torch.save(data, "data_1D_20N_10Gaussian.pt")

if __name__ == "__main__":
    gen_1D_20N_10Gaussian()


    