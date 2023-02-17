import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from random import choice
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from env.environnement import Env
from agents.agent import Agent

from torch.distributions import Categorical


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()


def transform_step_num(step_num:int):
    one_hot_it = np.zeros(500)
    one_hot_it[:step_num+1] = 1
    return one_hot_it

def state_to_obs(state):
    one_hotted_step_num = transform_step_num(state.step_num)
    #np_obs = np.hstack([state.map.reshape(-1),state.step_num])
    np_obs = np.concatenate([state.map.reshape(-1),one_hotted_step_num])
    return torch.tensor(np_obs, device=device, dtype=torch.float32).unsqueeze(0)


class Policy(nn.Module):
    def __init__(self, s_size, a_size, device):
        super(Policy, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(s_size, 1024, dtype=torch.float32, device=self.device)
        self.fc2 = nn.Linear(1024, a_size, dtype=torch.float32, device=self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)
    
    def act(self, state):
        #state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.forward(state.to(self.device))
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

class Policy_conv(nn.Module):

    def __init__(self, n_observations, n_actions,device):
        super(Policy_conv, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(1,4,kernel_size=3,stride=1,padding=0,dtype=torch.float32,device=self.device)
        self.conv2 = nn.Conv2d(5,20,kernel_size=3,stride=1,padding=0,dtype=torch.float32,device=self.device)
        self.dense1 = nn.Linear(3000,2048,dtype=torch.float32,device=self.device)
        self.dense2 = nn.Linear(2048, 512,dtype=torch.float32,device=self.device)
        self.dense3 = nn.Linear(512, 64,dtype=torch.float32,device=self.device)
        self.dense4 = nn.Linear(64, n_actions,dtype=torch.float32,device=self.device)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, state):
        x = state[:,:100]
        step_num = state[:,100:]
        step_num = torch.reshape(step_num,(-1,500))
        x1 = torch.reshape(x,(-1,10,10))
        new_x1 = torch.zeros([x1.shape[0],1,12, 12], dtype=torch.float32, device=self.device)
        new_x1[:,0, 1:11,1:11] = x1
        new_x1[:,0,0,1:11] = x1[:,9,:]
        new_x1[:,0,11,1:11] = x1[:,0,:]
        new_x1[:,0,1:11,0] = x1[:,:,9]
        new_x1[:,0,1:11,11] = x1[:,:,0]
        x2 = F.relu(self.conv1(new_x1))
        new_x = torch.zeros([x2.shape[0],4,12, 12], dtype=torch.float32, device=self.device)
        new_x[:,:, 1:11,1:11] = x2
        new_x[:,:,0,1:11] = x2[:,:,9,:]
        new_x[:,:,11,1:11] = x2[:,:,0,:]
        new_x[:,:,1:11,0] = x2[:,:,:,9]
        new_x[:,:,1:11,11] = x2[:,:,:,0]
        new_x = torch.concatenate([new_x,new_x1],dim=1)
        x = F.relu(self.conv2(new_x))
        x = torch.concatenate([x,torch.reshape(x1,(-1,1,10,10)),torch.reshape(x2,(-1,4,10,10))],dim=1)
        x = torch.reshape(x,(-1,2500))
        x = torch.concatenate([x,step_num],dim=1)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        return F.softmax(self.dense4(x),dim=1)

    def act(self, state):
        #state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state.to(self.device))
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


class Reinforce_Agent(Agent):
    def __init__(self,env:Env) -> None:
        # self.BATCH_SIZE = 64
        self.GAMMA = 0.9
        # self.EPS_START = 0.1
        # self.EPS_END = 0.00
        # self.EPS_DECAY = 15000
        # self.TAU = 0.002
        self.LR = 3e-6
    
        self.env = env
        self.n_actions = len(env.getPossibleActions())
        
        self.n_observations = 600
        self.policy = Policy_conv(self.n_observations, self.n_actions,device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.LR, amsgrad=True)

        self.saved_log_probs = []
        self.rewards = []
        self.scores_deque = deque(maxlen=100)
        self.scores = []
        self.training = True
        self.done = False

        self.rewards_to_display = []


    def act(self, state, training = True):
        self.training = training
        obs = state_to_obs(state)
        action, log_probs = self.policy.act(obs)
        self.saved_log_probs.append(log_probs)
        return action

    def observe(self, state, action, reward, next_state, done):
        if self.training:
            reward = torch.tensor([reward], device=device)
            self.rewards.append(reward)
            if done:
                self.done = True
                self.rewards_to_display.append(self.env.score)
                self.plot_rewards()
    
    def learn(self):
        if self.training:
            if self.done:
                
                self.scores_deque.append(sum(self.rewards))
                self.scores.append(sum(self.rewards))
                returns = deque(maxlen=self.env.max_step) 
                n_steps = len(self.rewards)
                for t in range(n_steps)[::-1]:
                    disc_return_t = (returns[0] if len(returns)>0 else 0)
                    returns.appendleft(self.GAMMA*disc_return_t + self.rewards[t])
                eps = np.finfo(np.float32).eps.item()
                returns = torch.tensor(returns, dtype=torch.float)
                returns = (returns - returns.mean()) / (returns.std() + eps)
                
                policy_loss = []
                for log_prob, disc_return in zip(self.saved_log_probs, returns):
                    policy_loss.append(-log_prob * disc_return)
                policy_loss = torch.cat(policy_loss).sum()

                self.optimizer.zero_grad()
                policy_loss.backward()
                self.optimizer.step()


                self.saved_log_probs = []
                self.rewards = []
                self.done = False

    def plot_rewards(self,show_result=False):
        plt.figure(1)
        rewards_t = torch.tensor(self.rewards_to_display, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(rewards_t.numpy())
        # Take 10 episode averages and plot them too
        if len(rewards_t) >= 10:
            means = rewards_t.unfold(0, 10, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(9), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())
