import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

#finite action DQN, after we place 20 gaussians, we reset the environment.

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
    
    def forward(self, x):
        x = x.reshape(-1, self.state_size)
        #print("Input shape:", x.shape)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size, N=100, gamma=0.99, epsilon=0.9, epsilon_decay=0.995, epsilon_min=0.05, learning_rate=1e-4, batch_size=32, max_action = 20):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.memory = []
        self.model = DQN(state_size, action_size) #the model we are training
        self.target_model = DQN(state_size, action_size) #the model we are using to predict the Q values
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.max_action = max_action
        self.action_counter = 0
        self.N = N
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.int64)  # Change dtype to float32
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32)

        Q_targets = rewards + self.gamma * torch.max(self.target_model(next_states), dim=1)[0] * (1 - dones)
        Q_targets = Q_targets.unsqueeze(1).detach()

        Q_values = self.model(states)
        
        # Gather Q-values using the individual action components
        action_positions = actions[:, 0].long().unsqueeze(1)
        action_heights = actions[:, 1]
        action_widths = actions[:, 2]
        
        Q_values_pos = torch.gather(Q_values[:, 0:self.N], 1, action_positions)  # Gather position component
        Q_values_height = torch.gather(Q_values[:, self.N:(self.N+5)], 1, action_heights)  # Gather height component
        Q_values_width = torch.gather(Q_values[:, (self.N+5):], 1, action_widths)  # Gather width component
        
        # Combine the gathered Q-values for the final Q-values tensor
        Q_values = torch.cat((Q_values_pos, Q_values_height, Q_values_width), dim=1)
        
        loss = self.criterion(Q_values, Q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def get_action(self, state):
        if self.action_counter >= self.max_action:
            return None

        if np.random.rand() <= self.epsilon:  # explore
            self.action_counter += 1
            action_position = np.random.randint(0, int(np.sqrt(self.state_size)))
            action_height = np.random.uniform(0.1, 5)
            action_width = np.random.uniform(0.1, 2)
            action = (action_position, action_height, action_width)
            return action
        
        #else: exploit
        self.action_counter += 1
        state = torch.tensor(state, dtype=torch.float32)
        Q_values = self.model(state)
        action = torch.argmax(Q_values).item()

        #convert action to gaussian parameters
        action_position = action // (10 * 5)
        action_height_idx = (action % (10 * 5)) // 5
        action_width_idx = (action % (10 * 5)) % 5
        action_height = np.linspace(0.1, 5, 10)[action_height_idx]
        action_width = np.linspace(0.1, 2, 5)[action_width_idx]

        action = (action_position, action_height, action_width)

        return action
    
    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)

    def reset_action_counter(self): #add a method to reset action counter at the end of each episode
        self.action_counter = 0
