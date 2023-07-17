from util import *
from posM_dqn import *
from env import *
import matplotlib.pyplot as plt
import numpy as np

#parameters
N = 100
kT = 0.596
state_start = 1
state_end = 88
num_gaussians = 10
time_step = 0.01
simulation_steps = 1000

state_size = ([1,], [N, N]) #position + M, shape in ([1, N, N])
action_size = [N] #we use a vector of length N to represent the number of gaussians at each position

set_randomseed(1)

#initialize the environment
env = posM_env_1d(N = N, 
                  kT = kT, 
                  state_start = state_start, 
                  state_end = state_end,
                  num_gaussians = num_gaussians,
                  time_step=time_step,
                  simulation_steps=simulation_steps,)

#initialize the agent
agent = pos_M_DQNAgent(state_size = state_size,
                       action_size = action_size,
                       num_gaussian=num_gaussians)

#we use the training defined in agent model.

num_episodes = 1000
reward = agent.train(env, num_episodes)

#plot the reward.
plt.plot(np.arange(len(reward)), reward)
plt.ylabel('reward')
plt.xlabel('episode')
plt.show()  

print("All Done!")


#here we load the trained model and test it.
agent.model.load_state_dict(torch.load('pos_M_DQN_model_13July_2_N20.pt'))
state = env.reset()
action = agent.get_action(state, train=False)

total_bias = np.zeros(N)
for position, num_gaussians in enumerate(action):
    total_bias += num_gaussians * gaussian(np.linspace(0, N, N), a=0.5, b=position, c=0.5)

K = create_K_1D(N)
F0 = compute_free_energy(K, kT)[1]
K_biased = bias_K_1D(K, total_bias, kT)

F = compute_free_energy(K_biased, kT)[1]
plt.plot(np.linspace(0, N - 1, N), F- F.min(), label='biased')
plt.plot(np.linspace(0, N - 1, N), F0 - F0.min(), label='unbiased')
plt.legend()
plt.show()