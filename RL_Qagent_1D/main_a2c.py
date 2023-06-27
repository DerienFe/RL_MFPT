from util import *
from a2c_agent import * # Import the A2C agent
from env import *
import matplotlib.pyplot as plt
import numpy as np

#parameters
N = 100
kT = 0.5981
state_start = 8
state_end = 89
max_action = 10
state_size = N*N
action_size = N

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

env = All_known_1D(N = N, kT = kT, state_start = state_start, state_end = state_end)
agent = A2CAgent(state_size, action_size, learning_rate=1e-3, max_action=max_action)

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
                done = agent.action_counter >= agent.max_action
                agent.update(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state
            
        ep_mfpt = env.get_mfpt(state)
        total_rewards.append(total_reward)
        mfpts.append(ep_mfpt)
        print(f"Episode: {episode+1}, \tTotal Reward: {total_reward:.2f}, \tMFPT: {ep_mfpt:.2f}")
        
        if episode in [100, 200, 400]:
            torch.save(agent.model.state_dict(), f'./model_26thJune{episode+1}_N100.pt')

num_episodes = 2500
train(num_episodes)
torch.save(agent.model.state_dict(), './model_26thJune_final_N100.pt')
print("Training done!")
np.savetxt('total_rewards.txt', total_rewards)

episodes = range(1, num_episodes + 1)
fig, ax1 = plt.subplots()
ax1.plot(episodes, total_rewards, 'b-')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Total Reward', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.plot(episodes, mfpts, 'r-')
ax2.set_ylabel('MFPT', color='r')
ax2.tick_params('y', colors='r')

plt.title('Total Reward and MFPT of Python Learn')
plt.show()
plt.savefig('RL_Qagent_1D_singleaction_0_N100.png')

def simulate():
    state = env.reset()
    agent.reset_action_counter()
    for _ in range(max_action):
        action = agent.get_action(state)
        print(action)
        state, reward = env.step(state, action)
    env.render(state)
    plt.show()
    plt.savefig('RL_Qagent_1D_sim_N100.png')

for _ in range(10):
    simulate()
print("Simulation done!")
