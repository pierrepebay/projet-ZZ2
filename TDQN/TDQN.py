import TdEnv
import Constants

import gym
import math
import random
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import yfinance as yf
from artemis.plotting.db_plotting import dbplot

from alpha_vantage.timeseries import TimeSeries
from pprint import pprint

app_key = Constants.APP_KEY

numberOfNeurons = 512
dropout = 0.2

ts = TimeSeries(key=app_key, output_format='pandas')
data, meta_data = ts.get_intraday(symbol='AAPL',interval='1min', outputsize='full')
dict = {"1. open": "Open","2. high": "High", "3. low": "Low", "4. close": "Close", "5. volume": "Volume"}
data.rename(columns=dict,
          inplace=True)

# # Load the stock data from a file and create the environment
# data = np.loadtxt('test.csv', delimiter=',')
# data=pd.read_csv("AAPL_stock_sample/AAPL_1hour_sample.txt", sep=",", header=None, names=["DateTime", "Open", "High", "Low", "Close", "Volume"])

# aapl = yf.Ticker("AAPL")
# data = aapl.history(period="1y", interval="1d")


env = TdEnv.TdEnv(data, 10000)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


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

    def __init__(self, n_observations, n_actions, numberOfNeurons=numberOfNeurons, dropout=dropout):
        super(DQN, self).__init__()
        # Definition of some Fully Connected layers
        self.fc1 = nn.Linear(n_observations, numberOfNeurons)
        self.fc2 = nn.Linear(numberOfNeurons, numberOfNeurons)
        self.fc3 = nn.Linear(numberOfNeurons, numberOfNeurons)
        self.fc4 = nn.Linear(numberOfNeurons, numberOfNeurons)
        self.fc5 = nn.Linear(numberOfNeurons, n_actions)

        # Definition of some Batch Normalization layers
        self.bn1 = nn.BatchNorm1d(numberOfNeurons)
        self.bn2 = nn.BatchNorm1d(numberOfNeurons)
        self.bn3 = nn.BatchNorm1d(numberOfNeurons)
        self.bn4 = nn.BatchNorm1d(numberOfNeurons)

        # Definition of some Dropout layers.
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        # Xavier initialization for the entire neural network
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.xavier_uniform_(self.fc4.weight)
        torch.nn.init.xavier_uniform_(self.fc5.weight)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, input):
        x = self.dropout1(F.leaky_relu(self.bn1(self.fc1(input))))
        x = self.dropout2(F.leaky_relu(self.bn2(self.fc2(x))))
        x = self.dropout3(F.leaky_relu(self.bn3(self.fc3(x))))
        x = self.dropout4(F.leaky_relu(self.bn4(self.fc4(x))))
        output = self.fc5(x)
        return output

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the AdamW optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
if gym.__version__[:4] == '0.26':
    state = env.reset()
elif gym.__version__[:4] == '0.25':
    state = env.reset(return_info=True)
n_observations = len([item for sublist in state for item in sublist])
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
policy_net.eval()
target_net.eval()
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations():
    plt.figure(1)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch to transition of batch-arrays.
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
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if torch.cuda.is_available():
    num_episodes = 6000
else:
    num_episodes = 50

cash_list = []
reward_list = []

for i_episode in range(num_episodes):
    print("\n##### Episode number {} #####\n".format(i_episode))

    # Initialize the environment and get it's state
    state = env.reset()
    state = torch.tensor([item for sublist in state for item in sublist], dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        policy_net.train()
        policy_net.eval()
        action = select_action(state)
        observation, reward, terminated, truncated = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated
        policy_net.train()

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor([item for sublist in observation for item in sublist], dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)
        if done:
            cash_list.append(env.getCash())
            print("Cash = {}, Shares = {}, Holdings = {}\n\n".format(env.getCash(), env.getNShares(), env.getHoldings()))
            print("Bought {} times, sold {} times.\n\n".format(env.nbought, env.nsold))
            episode_durations.append(env.getCash())
            dbplot(env.getMoney(), "money")
            #dbplot(env.getReturns(), "returns")
            #plot_durations()
            break
plt.plot(cash_list)
plt.show()
print('Complete')
plt.show()