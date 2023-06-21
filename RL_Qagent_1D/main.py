from util import *
from dqn_agent import *
from env import *

#parameters
N = 100
kT = 0.5981
state_start = 1
state_end = 18

state_size = N*N #because K shape is [N, N]
action_size = N*5*5 #because we have N grid points to put the gaussian

random.seed(2)
np.random.seed(2)
torch.manual_seed(2)

#initialize the environment
env = All_known_1D(N = N, kT = kT, state_start = state_start, state_end = state_end)

#initialize the agent
agent = DQNAgent(state_size, action_size, learning_rate=1e-3, max_action=10)

#def training.
num_episodes = 1000
def train(num_episodes):
    for episode in range(num_episodes):
        state = env.reset()
        agent.reset_action_counter()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.get_action(state)
            if action is not None:
                next_state, reward = env.step(state, action)
                total_reward += reward
                agent.remember(state, action, reward, next_state, done)
                agent.replay()
                state = next_state
                #print(f"Episode: {episode+1}, \tThis reward: {reward:.2f}, \tTotal Reward: {total_reward:.2f}")
            
            else: #else, we have placed 20 gaussians already, so we are done.
                done = True
            #env.render(state)
        ep_mfpt = env.get_mfpt(state)
        print(f"Episode: {episode+1}, \tTotal Reward: {total_reward:.2f}, \tMFPT: {ep_mfpt:.2f}")
        agent.decay_epsilon()
        agent.update_target_model()

#train(num_episodes)
#torch.save(agent.model.state_dict(), './RL_Qagent_1D/model_1.pt')

#now we split the training into different parts, so that we can save the model after each part.
# models are saved as model_1.pt, model_2.pt, etc.

num_episodes = 1000
for i in range(0, 10):
    #if there's a previous model, load it.
    if i > 0:
        agent.model.load_state_dict(torch.load(f'./RL_Qagent_1D/model2_abc_{i-1}.pt'))
    train(num_episodes)
    torch.save(agent.model.state_dict(), f'./RL_Qagent_1D/model2_abc_{i}.pt')
    print(f"Model {i+1} saved.")


#to load the model:
#agent.model.load_state_dict(torch.load('model_1.pt'))
print("Training done!")


#here we visualize the trained agent picking up 20 gaussians.
agent.model.load_state_dict(torch.load('./RL_Qagent_1D/model2_abc_4.pt'))

state = env.reset()
env.render(state)
agent.reset_action_counter()
for _ in range(10):
    # Select action from the agent
    action = agent.get_action(state)
    print(action)
    # Apply the action to the environment
    state, reward = env.step(state, action)

    # Plot the FES
env.render(state)
plt.show()