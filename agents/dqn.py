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

import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()
Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 1024)
        self.layer3 = nn.Linear(1024, 512)
        self.layer4 = nn.Linear(512, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)

class DQN_conv(nn.Module):

    def __init__(self, n_observations, n_actions,device):
        super(DQN_conv, self).__init__()
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
        return self.dense4(x)

def transform_step_num(step_num:int):
    one_hot_it = np.zeros(500)
    one_hot_it[:step_num+1] = 1
    return one_hot_it

def state_to_obs(state):
    one_hotted_step_num = transform_step_num(state.step_num)
    #np_obs = np.hstack([state.map.reshape(-1),state.step_num])
    np_obs = np.concatenate([state.map.reshape(-1),one_hotted_step_num])
    return torch.tensor(np_obs, device=device, dtype=torch.float32).unsqueeze(0)


class DQN_Agent(Agent):
    def __init__(self,env:Env) -> None:
        self.BATCH_SIZE = 64
        self.GAMMA = 1
        self.EPS_START = 0.1
        self.EPS_END = 0.00
        self.EPS_DECAY = 15000
        self.TAU = 0.002
        self.LR = 4e-4
        # Get number of actions from gym action space
        self.n_actions = len(env.getPossibleActions())
        self.n_observations = 101
        self.policy_net = DQN_conv(self.n_observations, self.n_actions,device=device).to(device)
        self.target_net = DQN_conv(self.n_observations, self.n_actions,device=device).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(10000)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=1, gamma=0.975)
        self.steps_done = 0

        self.rewards = []

        self.obs = None
        self.action = None
        self.next_obs = None
        self.reward = None

        self.env = env

        self.training = False
    

    def act(self,state, training = True):
        self.training = training
        self.obs = state_to_obs(state)
        self.action = self.select_action(self.obs)
        return self.action.item()

    def observe(self,state, action, reward, next_state, done):
        if self.training:
            self.reward = torch.tensor([reward-94], device=device)
            if done:
                print(state)
                print(self.env.score)
                self.next_obs = None
                print(self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY))
                self.rewards.append(self.env.score)
                self.plot_rewards()
                self.scheduler.step()
            else:
                self.next_obs = state_to_obs(next_state)

    def learn(self):
        if self.training:
            self.memory.push(self.obs,self.action,self.next_obs,self.reward)
            self.optimize_model()
            self.target_net_state_dict = self.target_net.state_dict()
            self.policy_net_state_dict = self.policy_net.state_dict()
            for key in self.policy_net_state_dict:
                self.target_net_state_dict[key] = self.policy_net_state_dict[key]*self.TAU + self.target_net_state_dict[key]*(1-self.TAU)
            self.target_net.load_state_dict(self.target_net_state_dict)

    def select_action(self,state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[choice(self.env.getPossibleActionsAsInt())]], device=device, dtype=torch.long)


    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
    
    def plot_rewards(self,show_result=False):
        plt.figure(1)
        rewards_t = torch.tensor(self.rewards, dtype=torch.float)
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
