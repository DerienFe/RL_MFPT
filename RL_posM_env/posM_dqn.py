# here we define a new dqn agent. using a parallel network combining the feature
#  of position [N] and feature of M [N,N] to predict the Q value.
# we also implement a parallel network to cat the two features together.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

from util import *

class pos_M_DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(pos_M_DQN, self).__init__()
        self.fc_pos1 = nn.Linear(state_size[0][0], 64)
        self.fc_pos2 = nn.Linear(64, hidden_size)
        self.fc_pos3 = nn.Linear(hidden_size, action_size)

        self.fc_M1 = nn.Linear(state_size[1][0], 32)
        self.fc_M2 = nn.Linear(32, 64)
        self.fc_M3 = nn.Linear(64, hidden_size)
        self.fc_M4 = nn.Linear(hidden_size, 64)
        self.fc_M5 = nn.Linear(64, action_size) #action_size = num_gaussians
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
    
    def forward(self, x):
        pos, M = x  # unpack state
        N = self.state_size[1][0]
        pos = pos.reshape(-1, 1).float() #pos in shape [batch_size, 1]
        M = M.reshape(-1, N, N).float()#M in shape [batch_size, N, N]

        # Extract band around the diagonal to be done.

        xpos = F.relu(self.fc_pos1(pos))
        xpos = F.relu(self.fc_pos2(xpos))
        xpos = self.fc_pos3(xpos) #xpos in shape [batch_size, action_size]

        xM = F.relu(self.fc_M1(M))
        xM = F.relu(self.fc_M2(xM))
        xM = F.relu(self.fc_M3(xM))
        xM = F.relu(self.fc_M4(xM))
        xM = self.fc_M5(xM) #xM in shape [batch_size, action_size]

        #we take average of the two outputs.
        x = (xpos + xM)/2
        return x

class pos_M_DQNAgent:
    def __init__(self, state_size, action_size, num_gaussian, gamma=0.99, epsilon=0.9, epsilon_decay=0.995, epsilon_min=0.05, learning_rate=1e-3, batch_size=32):
        self.state_size = state_size # state in shape ([1,], [N, N])
        self.action_size = action_size # action in shape [N]
        self.num_gaussian = num_gaussian

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        self.memory = []
        self.model = pos_M_DQN(self.state_size, action_size) #the model we are training
        self.target_model = pos_M_DQN(self.state_size, action_size) #the model we are using to predict the Q values
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def get_action(self, state):
        pos, M = state
        if np.random.rand() <= self.epsilon: #explore
            #gaussian params is a random sampling on [0, N-1], repeated num_gaussians times
            rng = np.random.default_rng()
            N = self.state_size[1][0]
            gaussian_params = rng.choice(np.linspace(0, N-1, N), size = self.num_gaussian, replace = True)
            return gaussian_params
        
        #else: exploit
        pos = torch.tensor(pos, dtype=torch.float32)
        M = torch.tensor(M.reshape(-1), dtype=torch.float64)
        state = (pos, M)
        Q_values = self.model(state)
        action = torch.argmax(Q_values).item()
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        pos_batch = torch.tensor([x[0][0] for x in batch], dtype=torch.float32)
        M_batch = torch.tensor([x[0][1] for x in batch], dtype=torch.float32)
        action_batch = torch.tensor([x[1] for x in batch], dtype=torch.int64)
        reward_batch = torch.tensor([x[2] for x in batch], dtype=torch.float32)
        next_pos_batch = torch.tensor([x[3][0] for x in batch], dtype=torch.float32)
        next_M_batch = torch.tensor([x[3][1] for x in batch], dtype=torch.float32)
        done_batch = torch.tensor([x[4] for x in batch], dtype=torch.bool)

        Q_values = self.model(pos_batch, M_batch)
        next_Q_values = self.target_model(next_pos_batch, next_M_batch)
        Q_value = Q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
        next_Q_value = next_Q_values.max(1)[0]
        expected_Q_value = reward_batch + self.gamma * next_Q_value * (1 - done_batch)
        
        loss = self.criterion(Q_value, expected_Q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def target_train(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train(self, env, episodes):
        rewards = [] #record rewards for each episode
        for e in range(episodes):
            # Initialize environment and state
            state = env.reset()
            done = False
            while not done:
                action = self.get_action(state)
                next_state, reward, done = env.step(state, action)
                self.remember(state, action, reward, next_state, done)
                self.replay()       # internally iterates default (prediction) model
                self.target_train() # iterates target model
                self.update_epsilon() # update exploration rate
                
                #update state
                state = next_state
            
            rewards.append(reward)

            if e % 10 == 0:
                print(f"episode: {e}/{episodes}, score: {reward}, e: {self.epsilon:.2}")

            torch.save(self.model.state_dict(), 'pos_M_DQN_model_7thJuly_1.pt')
        return rewards



    

