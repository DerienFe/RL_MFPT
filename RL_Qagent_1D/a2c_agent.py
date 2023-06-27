#this is a A2C agent

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=1)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value


class A2CAgent:
    def __init__(self, state_size, action_size, gamma=0.99, learning_rate=1e-4, max_action = 20):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.max_action = max_action
        self.action_counter = 0
        self.model = ActorCritic(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def get_action(self, state):
        if self.action_counter >= self.max_action:
            return None

        self.action_counter += 1
        state = torch.tensor(state, dtype=torch.float32)
        policy, _ = self.model(state)
        action = torch.distributions.Categorical(policy).sample().item()
        return action

    def update(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        _, value = self.model(state)
        _, next_value = self.model(next_state)

        target = reward + self.gamma * next_value * (1 - done)
        td_error = target - value

        log_policy, _ = self.model(state)
        log_policy = log_policy.squeeze(0)[action]
        actor_loss = -log_policy * td_error.detach()

        critic_loss = self.criterion(value, target.detach())

        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def reset_action_counter(self): #add a method to reset action counter at the end of each episode
        self.action_counter = 0
