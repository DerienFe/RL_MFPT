#this is a environment class for the 1D model.
# States are a representation of the current world or environment of the task. 
#  In our case, the state is the FES/K matrix.
# Actions are something an RL agent can do to change these states. 
#  In our case, where to put the gaussian.
# And rewards are the utility the agent receives for performing the “right” actions.
#  In our case, the reward is the biased mfpt from start state to end state.

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
        self.a = 2
        self.c = 0.75
        self.state_start = state_start
        self.state_end = state_end
        self.init_mfpt = self.get_mfpt(self.define_states())

    #initialize the K as state
    def define_states(self):
        states = create_K_1D(self.N)
        #self.render(states)
        return states
    
    #define the action space, adding a standard gaussian at the grid point from 0 to N.
    def define_actions(self):
        actions = np.array([i for i in range(self.N)])
        return actions
    
    def normalize_R(self, value, min_value, max_value):
        return 2 * ((value - min_value) / (max_value - min_value)) - 1

    #define the reward function
    def define_reward(self, state, action_taken):
        #state is the K matrix
        #action is the position of the gaussian
        #reward is the biased mfpt from start state to end state.
        #ts = 0.01 #time step， used in Mmfpt calculation.
        K = state

        #based on the action we create the bias, just a gaussian at the action position
        gaussian_bias = gaussian(np.linspace(0, self.N, self.N), a=self.a, b=action_taken, c=self.c)
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

        #change_percentage = (self.init_mfpt - mfpt_biased) / self.init_mfpt
        
        #reward = self.normalize_R(change_percentage, -0.5, 0.5)
    
        return (1.0/(mfpt_biased + 1e-6)) #negative reward, so that the agent will try to minimize the mfpt.
    
    #define the transition function
    def define_transition(self, state, action_taken):
        #state is the K matrix
        #action is the position of the gaussian
        #transition is the new state after the action is taken.
        K = state
        #based on the action we create the bias, just a gaussian at the action position
        gaussian_bias = gaussian(np.linspace(0, self.N, self.N), a=self.a, b=action_taken, c=self.c)
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
        #plt.show()
        return None