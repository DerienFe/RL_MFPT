#this is a environment class for the 1D model.
# States are a representation of the current world or environment of the task. 
#  In our case, the state is the FES/K matrix.
# Actions are something an RL agent can do to change these states. 
#  In our case, where to put the gaussian.
# And rewards are the utility the agent receives for performing the “right” actions.
#  In our case, the reward is the biased mfpt from start state to end state.


#note now it's a one-step game:
#Given the current position and the Markov Matrix.
# spit out the best gaussian combination that can achieve better in the random simulation.

#during the RL model training, we initialize the system, we train it until it reaches the end state.
# note that this training is different to one-step game, because we need to keep track of the whole trajectory.
from util import *

class posM_env_1d:
    def __init__(self,
                 N = 100,
                 kT = 0.5981,
                 state_start = 8,
                 state_end = 89,
                 num_gaussians = 10,
                 time_step = 0.01,
                 simulation_steps = 10000,
                ):
        self.N = N
        self.kT = kT
        self.a = 2
        self.c = 0.75
        self.num_gaussians = num_gaussians
        self.state_start = state_start
        self.state_end = state_end
        self.time_step = time_step
        self.simulation_steps = simulation_steps
        #self.action_space = spaces.Box(low=0, high=self.N, shape=(self.num_gaussians,), dtype=np.float32)
        #self.observation_space = spaces.Box(low=0, high=self.N, shape=(self.N,), dtype=np.float32)
        

    #initialize state 
    #maybe try randomizing the initial state?
    def define_states(self):
        K = create_K_1D(self.N)
        M = expm(K * self.time_step)
        M = real_nonnegative(M)
        M = M / np.sum(M, axis=1)[:, None]
        states = [self.state_start, M]
        return states
    
    def define_actions(self):
        actions = np.zeros(self.N) #now action space is a vector of length N, representing number of gaussians at each position 
        return actions
    
    #define the reward function
    def define_reward(self, state, action_taken):
        #state is the current 'state' or position; plus the M matrix.
        #action is the position of the gaussian, num_gaussians times.
        #reward is -1 or distance of the current state to the end state.
        
        next_state, done = self.define_transition(state, action_taken)
        next_pos, next_M = next_state #unpack the state

        #now we have the new state, we calculate the reward.
        distance = np.abs(self.state_end - next_pos)
        reward = -1*distance

        print("reward is: ", reward)
        return reward
    
    #define the transition function
    def define_transition(self, state, action_taken):
        #state is the current 'state' or position; plus the M0 matrix.
        #action is the position of the gaussian, num_gaussians times.
        #reward is -1 or distance of the current state to the end state.
        pos, M = state #unpack the state
        
        gaussian_numbers = action_taken # we have N gaussian numbers, each represents the number of gaussians at a position.
        total_bias = np.zeros(self.N)
        
        # calculate the total bias
        for position, num_gaussians in enumerate(gaussian_numbers):
            total_bias += num_gaussians * gaussian(np.linspace(0, self.N, self.N), a=self.a, b=position, c=self.c)

        #now we have the bias, we update the M matrix.
        M = bias_M(M, total_bias, time_step=self.time_step) #or we keep track of K? it doesn't really matter. the game is one-step.

        #filter to make sure the M matrix is all positive and real.
        # then normalize M to sum to 1.
        M = M.real
        M = np.where(M < 0, 0, M)
        M = M / np.sum(M, axis=1)[:, None]
        
        #now we have the M matrix, we update the position.
        [traj, steps_used, pos, done] = simulate(self.state_start, self.state_end, M, steps = self.simulation_steps)

        #now we have the new position, we update the state.
        state = [pos, M]
        return (state, done)
    
    #define the step function

    def step(self, state, action_taken):
        #state is the K matrix
        #action is the position of the gaussian
        #step is the new state after the action is taken.
        #reward = self.define_reward(state, action_taken)
        next_state, done = self.define_transition(state, action_taken)
        distance = np.abs(self.state_end - next_state[0])
        reward = -1*distance
        return next_state, reward, done
    
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