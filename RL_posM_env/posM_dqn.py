# here we define a new dqn agent. using a parallel network combining the feature
#  of position [N] and feature of M [N,N] to predict the Q value.
# we also implement a parallel network to cat the two features together.
#state is [pos, M] #action is [N]
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
        self.fc_pos1 = nn.Linear(1, 64)
        self.fc_pos2 = nn.Linear(64, hidden_size)
        self.fc_pos3 = nn.Linear(hidden_size, action_size[0])

        self.conv1M = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2M = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2M_drop = nn.Dropout2d()
        self.fc1M = nn.Linear(40000, hidden_size) # depends on the size of M, N=20, 1600 N = 100, 40000
        self.fc2M = nn.Linear(hidden_size, action_size[0])

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size

    def forward(self, x):
        pos, M = x  # unpack state
        N = self.state_size[1][0]

        M = M.reshape(-1, 1, N, N) # reshape M to be [batch_size, 1, N, N]
        # Extract band around the diagonal to be done.

        xpos = F.relu(self.fc_pos1(pos.unsqueeze(0)))
        xpos = F.relu(self.fc_pos2(xpos))
        xpos = self.fc_pos3(xpos) #xpos in shape [batch_size, action_size]

        xM = F.relu(F.max_pool2d(self.conv1M(M), 2))
        xM = F.relu(F.max_pool2d(self.conv2M_drop(self.conv2M(xM)), 2))
        xM = xM.view(xM.size(0), -1) # flatten over channel, height and width = [batch_size,num_flat_features]
        
        xM = F.relu(self.fc1M(xM))
        xM = self.fc2M(xM) #xM in shape [batch_size, action_size]

        #we take average of the two outputs.
        x = (xpos + xM)/2
        return x

class pos_M_DQNAgent:
    def __init__(self, 
                 state_size, 
                 action_size, 
                 num_gaussian, 
                 gamma=0.99, 
                 epsilon=0.9, 
                 epsilon_decay=0.995, #0.995
                 epsilon_min=0.05, 
                 learning_rate=1e-3, 
                 batch_size=4):
        self.state_size = state_size # state in shape ([1,], [N, N]) 
        self.action_size = action_size # action in shape [num_gaussian]
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
        self.criterion = nn.SmoothL1Loss() #nn.MSELoss() #use L1 loss? MSE loss may behave bad.
        self.rng = np.random.default_rng()
        self.max_num_model_update = 4 # number of model update permitted for each episode. because we never reach the terminal state, we need to limit the number of model update.
        self.num_model_update = 0

    def get_action(self, state, train=True):
        pos, M = state
        N = self.state_size[1][0]
        #explore
        r = np.random.rand()
        if train == True:
            if r <= self.epsilon: #?or len(self.memory) < self.batch_size: 
                #gaussian params is a random sampling on [0, N-1], repeated num_gaussians times
                idx = self.rng.choice(np.linspace(0, N-1, N), size = self.num_gaussian, replace = True)
                #note our action is a list ranging from 0 to N-1, each non-zero value means the number of gaussian placed at the corresponding position.
                action = np.zeros(N)
                for i in idx:
                    action[int(i)] += 1
                print("EXPLORING, gaussians applied on:", idx)
                return action
            
            #else: exploit
            else:
                pos = torch.tensor(pos, dtype=torch.float32)
                M = torch.tensor(M, dtype=torch.float32)
                state = (pos, M)
                Q_values = self.model(state) #shape in [batch_size, action_space]
                _, topk_indices = torch.topk(Q_values, self.num_gaussian, dim=1)
                
                action = np.zeros(N)
                for i in topk_indices.squeeze().numpy():
                    action[i] += 1
                print("EXPLOITING, gaussians applied on:", topk_indices.squeeze())
                return action
        else:
            #use exploit only
            pos = torch.tensor(pos, dtype=torch.float32)
            M = torch.tensor(M, dtype=torch.float32)
            state = (pos, M)
            Q_values = self.model(state) #shape in [batch_size, action_space]
            _, topk_indices = torch.topk(Q_values, self.num_gaussian, dim=1)
                
            action = np.zeros(N)
            for i in topk_indices.squeeze().numpy():
                action[i] += 1
            print("EXPLOITING, gaussians applied on:", topk_indices.squeeze())
            return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        
        N = self.state_size[1][0]
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)

        #here we sample randomly from memory, and cat the tensors.
        pos_batch = torch.tensor(np.array([x[0][0] for x in batch]), dtype=torch.float32).unsqueeze(1)
        M_batch = torch.tensor(np.array([x[0][1] for x in batch]), dtype=torch.float32).reshape(-1, 1, N, N)
        action_batch = torch.tensor(np.array([x[1] for x in batch]), dtype=torch.float32)  # in shape [batch_size, N]
        reward_batch = torch.tensor(np.array([x[2] for x in batch]), dtype=torch.float32)

        next_pos_batch = torch.tensor(np.array([x[3][0] for x in batch]), dtype=torch.float32).unsqueeze(1)
        next_M_batch = torch.tensor(np.array([x[3][1] for x in batch]), dtype=torch.float32).reshape(-1, 1, N, N)
        done_batch = torch.tensor(np.array([x[4] for x in batch]), dtype=torch.bool)


        """
        Note, here the action_batch is in shape [batch_size, N], each row is a list of indices of the gaussians.
        however we can't gather it into our q values directly.
        To solve this we can multiply it by the corresponding q values and sum over 
        the action dimension. thus get a single q value for each 'set of action' in the batch.
        """
        Q_values = self.model((pos_batch, M_batch)) #shape in [batch_size, N]
        Q_values_selected = (Q_values.squeeze() * action_batch).sum(dim=1)
        
        with torch.no_grad():
            next_Q_values = self.target_model((next_pos_batch, next_M_batch)).squeeze() #shape in [batch_size, N]
            
        next_state_values = torch.zeros(self.batch_size)

        next_state_values[done_batch] = 0.0
        next_state_values[~done_batch] = next_Q_values.max(1)[0].detach()[~done_batch]

        expected_Q_values = reward_batch + (self.gamma * next_state_values)

        loss = self.criterion(Q_values_selected, expected_Q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.num_model_update += 1
    
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def target_train(self):
        self.target_model.load_state_dict(self.model.state_dict())


    def train(self, env, episodes):
        rewards = [] #record rewards for each episode
        for e in range(episodes):
            print("###########    Starting epoch: ", e,"    ###########")
            #set random seed
            set_randomseed(e)
            # Initialize environment and state
            state = env.reset()
            self.num_model_update = 0
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

                if self.num_model_update >= self.max_num_model_update:
                    break
            
            rewards.append(reward)

            if e % 10 == 0:
                print(f"episode: {e}/{episodes}, score: {reward}, e: {self.epsilon:.2}")

            if e % 100 == 0:
                torch.save(self.model.state_dict(), 'pos_M_DQN_model_14July_1_N20.pt')
        return rewards


