#now instead of using K matrix as the state, we use the position on x-axis as state.
#action is still the position of the gaussian.
# now action will place a gaussian on the x-axis
# transition will be the new position on the x-axis after the action is taken.
#  transition is decided by the corresponding element in the transition matrix M = expm(K * ts)
#  the new position is decided by random sampling from the probability distribution of the corresponding row in M.
#  thus it could be in 3 cases: left, right, stay.
# note that in transition, we propagate the system for sim_step times to get new state.
# reward is the distance between the new position and the end state.
#  the reward is negative, so that the agent will try to minimize the distance.


#note, i have to make sure the state is surficient to make the action.
#then where should i get information for the M?
from util import *
from scipy.linalg import expm

class position_1D:
    def __init__(self,
                 N = 100,
                 kT = 0.5981,
                 state_start = 8,
                 state_end = 89,
                 time_step = 0.01,
                 sim_step = 1000
                 ):
        self.N = N
        self.kT = kT
        self.state_start = state_start
        self.state_end = state_end
        self.K = self.init_K()
        self.time_step = time_step
        self.sim_step = int(sim_step)
    
    def init_K(self):
        return create_K_1D(self.N)

    #initialize the position on x-axis as state
    def define_states(self):
        return self.state_start
    
    #define the action space, adding a standard gaussian at the grid point from 0 to N.
    def define_actions(self):
        actions = np.array([i for i in range(self.N)])
        return actions
    
    def propagate(self, state, action_taken):
        record_state = []

    def transition(self, state, action_taken):
        gaussian_bias = gaussian(np.linspace(0, self.N, self.N), a=1, b=action_taken, c=0.5)
        bias_K = bias_K_1D(self.K, gaussian_bias)

        bias_M = expm(bias_K * self.time_step)
        #normalize the transition matrix. so that the probability of each row sums to 1.
        bias_M = bias_M / np.sum(bias_M, axis=1)[:, None]

        #propagate the system for sim_step times
        for i in range(self.sim_step):
            new_state = np.random.choice(np.array([i for i in range(self.N)]), p=bias_M[state, :])
        
        return new_state
    
    def reward(self, state, action_taken):
        new_state = self.transition(state, action_taken)
        reward = -abs(new_state - self.state_end)
        return reward
    
    def reset(self):
        state = self.define_states()
        return state
    
    def step(self, state, action_taken):
        reward = self.reward(state, action_taken)
        new_state = self.transition(state, action_taken)
        return new_state, reward
    
    
    


