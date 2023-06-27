from util import *
from dqn_agent import *
from env import *
import matplotlib.pyplot as plt
import numpy as np

#parameters
N = 20
kT = 0.5981
state_start = 2
state_end = 18
max_action = 10
state_size = N*N #because K shape is [N, N]
action_size = N #because we have N grid points to put the gaussian

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

#initialize the environment
env = All_known_1D(N = N, kT = kT, state_start = state_start, state_end = state_end)

#initialize the agent
agent = DQNAgent(state_size, action_size, learning_rate=1e-3, max_action=max_action)

#def training.
total_rewards = []
mfpts = []
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
        total_rewards.append(total_reward)
        mfpts.append(ep_mfpt)
        print(f"Episode: {episode+1}, \tTotal Reward: {total_reward:.2f}, \tMFPT: {ep_mfpt:.2f}")
        agent.decay_epsilon()
        agent.update_target_model()

        if episode == 200:
            torch.save(agent.model.state_dict(), f'./model_26thJune{i+1}_200.pt')
        elif episode == 400:
            torch.save(agent.model.state_dict(), f'./model_26thJune{i+1}_400.pt')
        elif episode == 100:
            torch.save(agent.model.state_dict(), f'./model_26thJune{i+1}_100.pt')
#train(num_episodes)
#torch.save(agent.model.state_dict(), './RL_Qagent_1D/model_1.pt')

#now we split the training into different parts, so that we can save the model after each part.
# models are saved as model_1.pt, model_2.pt, etc.

num_episodes = 500

for i in range(0,1):
    #if there's a previous model, load it.
    if i > 0:
        agent.model.load_state_dict(torch.load(f'./model_27thJune{i}_N20.pt'))
    train(num_episodes)
    torch.save(agent.model.state_dict(), f'./model_27thJune{i+1}_N20.pt')
    print(f"Model {i+1} saved.")

"""
#to load the model:
#agent.model.load_state_dict(torch.load('model_1.pt'))
print("Training done!")
np.savetxt('total_rewards.txt', total_rewards)
#here we visualize the trained agent picking up 20 gaussians.
#agent.model.load_state_dict(torch.load('./RL_Qagent_1D/model_1.pt'))

#plot the total reward during training.
episodes = range(1, num_episodes + 1)
fig, ax1 = plt.subplots()

# Plot total rewards
ax1.plot(episodes, total_rewards, 'b-')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Total Reward', color='b')
ax1.tick_params('y', colors='b')

# Create a second y-axis for MFPTs
ax2 = ax1.twinx()
ax2.plot(episodes, mfpts, 'r-')
ax2.set_ylabel('MFPT', color='r')
ax2.tick_params('y', colors='r')

plt.title('Total Reward and MFPT of Python Learn')
plt.show()
plt.savefig('RL_Qagent_1D_singleaction_0_N100.png')
"""

#here we def a simulate function that use the trained model to pick up 20 gaussians.

def simulate():
    state = env.reset()
    agent.reset_action_counter()

    #here's the initial FES
    env.render(state)

    for _ in range(max_action):
        # Select action from the agent
        action = agent.get_action(state)
        print(action)
        # Apply the action to the environment
        state, reward = env.step(state, action)
        env.render(state)
    plt.show()
    #plt.savefig('RL_Qagent_1D_sim_N100.png')

for _ in range(10):
    simulate()
    print("Simulation done!")

print("All done")