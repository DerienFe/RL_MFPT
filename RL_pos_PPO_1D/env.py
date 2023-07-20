#this is a environment class for the 1D model.
# States are a observation of the current world or environment of the task. 
#  In our case, the state is the position on the x-axis.
# Actions is a vector with shape [N], each element represents the number of gaussians at a position.
# And rewards is the distance of the current position to the end position.

#note now it's a one-step game:
#Given the current position
# perturb the original K matrix, get biased K, get biased M, propagate the system, get the new position.

#during the RL model training, we initialize the system, we train it until it reaches the end state.
# note that this training is different to one-step game, because we need to keep track of the whole trajectory.

#edit by TW, 19th July 2023.
from util import *
import gym
from gym import spaces
import numpy as np


class pos_1D_env(gym.Env):
    metadata = {'render_modes': ['human'], "render_fps": 6}

    def __init__(self,
                 N = 100,
                 kT = 0.5981,
                 state_start = 8,
                 state_end = 89,
                 num_gaussians = 10,
                 time_step = 0.1,
                 simulation_steps = 10000,
                 render_mode = None,
                ):
        self.N = N
        self.kT = kT
        self.a = 1
        self.c = 0.5
        self.num_gaussians = num_gaussians
        self.state_start = state_start
        self.state_end = state_end
        self.time_step = time_step
        self.simulation_steps = simulation_steps
        self.K = create_K_1D(self.N)

        self.observation_space = spaces.Discrete(self.N)
        #define env.observation_space.n as N.
        self.observation_space.n = self.N
        self.action_space = spaces.MultiDiscrete([self.num_gaussians] * self.N) #MultiDiscrete([m] * n), it will represent n discrete actions, each with m possible values (from 0 to m-1)
        #define env.action_space.nvec as [num_gaussians] * self.N
        self.action_space.nvec = [self.num_gaussians] * self.N
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.max_simulation_steps = 1e7
        self.total_time_elapsed = 0
        self.terminated = False

        self.global_info = None
        self.episodic_info = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = self.state_start

        #convert to int64 to avoid overflow
        observation = int(self.state)
        info = {"trajectory": [], 
                "step": 0, 
                "current_position": int(self.state), 
                "done": False,
                }
        
        self.global_info = {"global_trajectory": [],
                        "global_step": 0,
                        "global_current_position": int(self.state),
                        "global_done": False,
                        }
        
        self.episodic_info = info
        self.episodic_info["last_action"] = None

        return observation, info

    def step(self, action):
        """
        here we use the functions from util.py. propagate the system given the action.
        """
        
        #get total bias from the action.
        total_bias = np.zeros((self.N))
        for position, num_gaussians in enumerate(action):
            total_bias += num_gaussians * gaussian(np.linspace(0, self.N, self.N), a=self.a, b=position, c=self.c)

        #now we have the bias, we get biased_K, biased_M and propagate the system.
        biased_K = bias_K_1D(self.K, total_bias)
        biased_M = expm(biased_K * self.time_step).T #the K in python is the transpose of K in matlab. thus to use the correct M, we have to transpose it.
        biased_M = real_nonnegative(biased_M)
        biased_M = biased_M / np.sum(biased_M, axis=1)[:, None]

        traj, step, self.state, done = simulate(self.state, self.state_end, biased_M, self.simulation_steps)


        #recordings.
        observation = int(self.state)
        self.total_time_elapsed += step

        reward = -1 * np.abs(self.state_end - self.state)

        info = {"trajectory": traj,
                "step": step,
                "current_position": self.state,
                "done": done,
                }
        
        self.global_info["global_trajectory"].append(traj) #a list.
        self.global_info["global_step"] += step
        self.global_info["global_current_position"] = self.state
        self.global_info["global_done"] = done

        self.episodic_info = info
        self.episodic_info["last_action"] = action

        #determine whether the game is overtime
        if self.total_time_elapsed > self.max_simulation_steps:
            self.terminated = True

        #return observation, reward, done, self.terminated, info
        return observation, reward, done, info
    
    def render(self, mode="human"):
        """
        here we use matplotlib to render the environment.
        """
        
        F = compute_free_energy(self.K, self.kT)[1]
        F = F - F.min()
        
        last_action = self.episodic_info["last_action"]
        if last_action is None:
            last_action = np.zeros(self.N)
        
        total_bias = np.zeros((self.N))
        for position, num_gaussians in enumerate(last_action):
            total_bias += num_gaussians * gaussian(np.linspace(0, self.N, self.N), a=self.a, b=position, c=self.c)
        
        biased_K = bias_K_1D(self.K, total_bias)
        biased_F = compute_free_energy(biased_K, self.kT)[1]
        biased_F = biased_F - biased_F.min()
        #here we plot the free energy landscape. BLUE
        # we also plot the biased free energy landscape with last action. GREEN
        # and the position of the system with a BLACK dot.
        # and the end position with a RED dot.
        # and the trajectory with scattered black dots with 25% transparency.
        # in the title we show the global steps used.
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(self.N), F, color='blue')
        plt.plot(np.arange(self.N), biased_F, color='green')
        plt.plot(self.state, F[self.state], 'ko')
        plt.plot(self.state_end, F[self.state_end], 'ro')
        plt.plot(self.episodic_info["trajectory"], F[self.episodic_info["trajectory"]], 'ko', alpha=0.25)
        plt.title("global steps used: " + str(self.global_info["global_step"]) + "  reward: " + str(-1 * np.abs(self.state_end - self.state)))
        plt.show()
        return None
