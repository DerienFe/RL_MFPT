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
simulation_steps = 10000

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

num_episodes = 100
agent.train(env, num_episodes)



