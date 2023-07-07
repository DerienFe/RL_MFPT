import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

class CQL(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(CQL, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
    
    def forward(self, x):
        x = x.reshape(-1, self.state_size)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CQLAgent:
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=0.9, epsilon_decay=0.995, epsilon_min=0.05, learning_rate=1e-4, batch_size=32, max_action = 20, alpha=1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.memory = []
        self.model = CQL(state_size, action_size) #the model we are training
        self.target_model = CQL(state_size, action_size) #the model we are using to predict the Q values
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.max_action = max_action
        self.action_counter = 0
        self.alpha = alpha # scaling factor for the log-sum-exp operation in CQL loss

    # ... remember and get_action methods are the same as in DQNAgent ...
        
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(np.array(states), dtype=torch.float64)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float64)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float64)
        dones = torch.tensor(np.array(dones), dtype=torch.bool)
        
        Q_targets = rewards + self.gamma * torch.max(self.target_model(next_states), dim=1)[0] * (1 - dones)
        Q_targets = Q_targets.detach()
        
        Q_values = self.model(states)
        Q_values_selected = torch.gather(Q_values, 1, actions.unsqueeze(1)).squeeze(1)
        
        # calculate CQL loss
        random_actions = torch.randint(self.action_size, size=(self.batch_size,)).to(Q_values.device)
        Q_values_random = torch.gather(Q_values, 1, random_actions.unsqueeze(1)).squeeze(1)
        log_sum_exp_Q = torch.logsumexp(Q_values / self.alpha, dim=1)
        cql_loss = self.alpha * (Q_values_selected - log_sum_exp_Q)
        
        # calculate total loss: MSE loss + CQL loss
        mse_loss = self.criterion(Q_values_selected, Q_targets)
        loss = mse_loss + cql_loss.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def get_action(self, state):
        if self.action_counter >= self.max_action:
            return None

        if np.random.rand() <= self.epsilon: #explore
            self.action_counter += 1
            return random.randint(0, self.action_size - 1)
        
        #else: exploit
        self.action_counter += 1
        state = torch.tensor(state, dtype=torch.float32)
        Q_values = self.model(state)
        action = torch.argmax(Q_values).item()
        return action
    
    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)

    def reset_action_counter(self): #add a method to reset action counter at the end of each episode
        self.action_counter = 0
