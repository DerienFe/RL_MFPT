#this is a environment class for the 1D model.
# States are a representation of the current world or environment of the task. 
#  In our case, the state is the FES/K matrix.
# Actions are something an RL agent can do to change these states. 
#  In our case, where to put the gaussian.
# And rewards are the utility the agent receives for performing the “right” actions.
#  In our case, the reward is the biased mfpt from start state to end state.

#to do: expand the action into 2 divided action:
# 1)select where to put gaussian
# 2)how high/wide the gaussian is.

from util import *
from scipy.linalg import expm

class All_known_1D:
    def __init__(self,
                 N = 100,
                 kT = 0.5981,
                 state_start = 8,
                 state_end = 89,
                ):
        self.N = N
        self.kT = kT
        self.state_start = state_start
        self.state_end = state_end
        self.init_mfpt = self.get_mfpt(self.define_states())

    #initialize the K as state
    def define_states(self):
        states = create_K_1D(self.N)
        #self.render(states)
        return states
    
    #define the action space, chose where to put the gaussian
    # then chose the height and width of the gaussian.
    def define_actions(self):
        action_space = []
        
        # Define the positions where the Gaussian can be placed
        positions = np.linspace(0, self.N - 1, self.N)
        
        # Define the range of heights and widths for the Gaussian
        height_range = np.linspace(0.1, 5, 5)  # Example height range
        width_range = np.linspace(0.1, 2, 5)  # Example width range
        
        # Generate all possible combinations of positions, heights, and widths
        for position in positions:
            for height in height_range:
                for width in width_range:
                    action_space.append((position, height, width))
        return action_space

    def normalize_R(self, value, min_value, max_value):
        return 2 * ((value - min_value) / (max_value - min_value)) - 1

    #define the reward function
    def define_reward(self, state, action_taken):
        #state is the K matrix
        #action is the position of the gaussian
        #reward is the biased mfpt from start state to end state.
        ts = 0.01 #time step， used in Mmfpt calculation.
        K = state
        action_position, action_height, action_width = action_taken #unpack values

        #based on the action we create the bias, place a gaussian
        gaussian_bias = gaussian(np.linspace(0, self.N, self.N), a=action_height, b=action_position, c=action_width)
        bias_K = bias_K_1D(K, gaussian_bias)

        #compute the mfpt
        peq, F, evectors, evalues, evalues_sorted, index = compute_free_energy(bias_K, self.kT)
        #mfpts = mfpt_calc(peq, K)
        
        mfpts_biased = mfpt_calc(peq, bias_K)

        #Mt = expm(bias_K * ts)
        #Mmfpts_biased = markov_mfpt_calc(peq, Mt)

        #mfpt = mfpts[self.state_start, self.state_end] #point of interest mfpt. Unbiased.
        mfpt_biased = mfpts_biased[self.state_start, self.state_end] #point of interest mfpt.
        #Mmfpt_biased = Mmfpts_biased[self.state_start, self.state_end] / ts#point of interest mfpt.

        """#reward the exploration.
        print("action taken is:", action_taken)
        self.action_count[state, int(action_taken)] += 1
        exploration_bonus = 1 / (1 + np.sqrt(self.action_count[state, action_taken]))
"""
        #if mfpt is smaller, then the reward is bigger.
        #quantify the reward as the percentate of the mfpt reduction.
        #set a threshold of reward to avoid the reward/punishment being too big.

        change_percentage = (self.init_mfpt - mfpt_biased) / self.init_mfpt
        
        #reward = self.normalize_R(change_percentage, -0.5, 0.5)
    
        return (change_percentage) #negative reward, so that the agent will try to minimize the mfpt.
    
    #define the transition function
    def define_transition(self, state, action_taken):
        #state is the K matrix
        #action is the position of the gaussian
        #transition is the new state after the action is taken.
        action_position, action_height, action_width = action_taken #unpack values

        K = state
        #based on the action we create the bias, just a gaussian at the action position
        gaussian_bias = gaussian(np.array([i for i in range(100)]), a=action_height, b=action_position, c=action_width)
        bias_K = bias_K_1D(K, gaussian_bias)
        #render the new state
        #self.render(bias_K)
        return bias_K
    
    def get_mfpt(self, state):
        K = state
        peq, F, evectors, evalues, evalues_sorted, index = compute_free_energy(K, self.kT)
        mfpts = mfpt_calc(peq, K)
        mfpt = mfpts[self.state_start, self.state_end]
        return mfpt

    #define the step function
    def step(self, state, action_taken):
        #state is the K matrix
        #action is the position of the gaussian
        #step is the new state after the action is taken.
        reward = self.define_reward(state, action_taken)
        transition = self.define_transition(state, action_taken)
        return transition, reward
    
    #define the reset function
    def reset(self):
        state = self.define_states()
        return state
    
    #define the render function
    def render(self, state):
        #state is the K matrix
        #render is the plot of the K matrix.
        K = state
        peq, F, evectors, evalues, evalues_sorted, index = compute_free_energy(K, self.kT)
        #print(F)
        #plot the free energy surface
        plt.figure(figsize=(10, 5))
        plt.plot(np.linspace(0, self.N - 1, self.N), F)
        plt.show()
        return None