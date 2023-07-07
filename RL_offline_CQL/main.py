#this is the main file for the offline CQL training.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from cql_agent import *

data = torch.load("../data/data_1D_20N_10Gaussian.pt")

# Separate the data into states, actions, rewards, and next_states
states = torch.stack([x[0] for x in data]).float()
#actions = torch.stack([x[1] for x in data]).long()
for x in data:
    x[1] = torch.tensor(x[1]).long()
    print(x[1])

rewards = torch.stack([x[2] for x in data]).float()
next_states = torch.stack([x[3] for x in data]).float()

data = TensorDataset(states, actions, rewards, next_states)
dataloader = DataLoader(data, batch_size=64, shuffle=True)

agent = CQL(state_dim = 20, action_dim = 20)
agent.train(data, epochs=100)
