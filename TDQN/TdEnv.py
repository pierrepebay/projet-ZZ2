import os
import gym
import math
import numpy as np

import pandas as pd
pd.options.mode.chained_assignment = None

from matplotlib import pyplot as plt

class TdEnv(gym.Env):

    def __init__(self, data, money, length=30,
                 transactionCosts=0):

        # If affirmative, load the stock market data from the database
        self.data = data

        # fix data
        self.data.replace(0.0, np.nan, inplace=True)
        self.data.interpolate(method='linear', limit=5, limit_area='inside', inplace=True)
        self.data.fillna(method='ffill', inplace=True)
        self.data.fillna(method='bfill', inplace=True)
        self.data.fillna(0, inplace=True)
        
        # Set the trading activity data
        self.dataLength = len(self.data['Close'])+length
        self.position = np.array([0 for _ in range(self.dataLength)])
        self.action = np.array([0 for _ in range(self.dataLength)])
        self.holdings = np.array([0. for _ in range(self.dataLength)])
        self.cash = np.array([float(money) for _ in range(self.dataLength)])
        self.money = np.array([0. + float(money) for _ in range(self.dataLength)])
        self.returns = np.array([0. for _ in range(self.dataLength)])
        self.boughtAndHolding = False
        self.soldAndHolding = True

        # Set info variables
        self.nbought = 0
        self.nsold = 0

        # Set the RL variables common to every OpenAI gym environments
        self.state = [self.data['Open'][0:length].tolist(),
                      self.data['High'][0:length].tolist(),
                      self.data['Low'][0:length].tolist(),
                      self.data['Close'][0:length].tolist(),
                      self.data['Volume'][0:length].tolist(),
                      [0]]
        self.reward = 0.
        self.done = 0
        self.action_space = gym.spaces.Discrete(3)

        # Set additional variables related to the trading activity
        self.length = length
        self.t = length
        self.numberOfShares = 0
        self.transactionCosts = transactionCosts
        self.epsilon = 0.1

    def getCash(self):
        return self.cash[self.t]
    
    def getNShares(self):
        return self.numberOfShares
    
    def getHoldings(self):
        return self.holdings[self.t]

    def getMoney(self):
        return self.money[self.t]
    
    def getReturns(self):
        return self.returns[self.t]
    
    def getPositionString(self):
        v = self.position[self.t]
        if v == 0:
            return "Hold"
        elif(v == 1):
            return "Bought"
        elif(v == -1):
            return "Sold"
        else:
            return "Something went wrong"

    def reset(self):

        # Reset the trading activity dataframe
        self.position = np.array([0 for _ in range(self.dataLength)])
        self.action = np.array([0 for _ in range(self.dataLength)])
        self.holdings = np.array([0. for _ in range(self.dataLength)])
        self.cash = np.array([float(self.cash[0]) for _ in range(self.dataLength)])
        self.money = np.array([0. + float(self.cash[0]) for _ in range(self.dataLength)])
        self.returns = np.array([0. for _ in range(self.dataLength)])
        self.boughtAndHolding = False
        self.soldAndHolding = True

        # Reset the RL variables common to every OpenAI gym environments
        self.state = [self.data['Open'][0:self.length].tolist(),
                      self.data['High'][0:self.length].tolist(),
                      self.data['Low'][0:self.length].tolist(),
                      self.data['Close'][0:self.length].tolist(),
                      self.data['Volume'][0:self.length].tolist(),
                      [0]]
        self.reward = 0.
        self.done = 0

        # Reset additional variables related to the trading activity
        self.t = self.length
        self.numberOfShares = 0

        self.nbought = 0
        self.nsold = 0

        return self.state

    
    def computeLowerBound(self, cash, numberOfShares, price):

        # Computation of the RL action lower bound
        deltaValues = - cash - numberOfShares * price * (1 + self.epsilon) * (1 + self.transactionCosts)
        if deltaValues < 0:
            lowerBound = deltaValues / (price * (2 * self.transactionCosts + (self.epsilon * (1 + self.transactionCosts))))
        else:
            lowerBound = deltaValues / (price * self.epsilon * (1 + self.transactionCosts))
        return lowerBound
    

    def step(self, action):

        t = self.t
        customReward = False

        # CASE 1: BUY
        if(action == 1 and self.soldAndHolding):
            # Case: No position -> Buy
            self.nbought += 1
            self.position[t + 1] = 1
            self.numberOfShares = math.floor(self.cash[t]/(self.data['Close'][t + 1] * (1 + self.transactionCosts)))
            self.cash[t + 1] = self.cash[t] - self.numberOfShares * self.data['Close'][t + 1] * (1 + self.transactionCosts)
            self.holdings[t + 1] = self.numberOfShares * self.data['Close'][t + 1]
            self.action[t + 1] = 1
            self.boughtAndHolding = True
            self.soldAndHolding = False

        # CASE 2: SELL (FROM BUY)
        elif(action == 0 and self.boughtAndHolding):
            # Case: Buy -> Sell
            self.nsold += 1
            self.position[t + 1] = -1
            self.cash[t + 1] = self.cash[t] + self.numberOfShares * self.data['Close'][t + 1] * (1 - self.transactionCosts)
            self.numberOfShares = math.floor(self.cash[t]/(self.data['Close'][t + 1] * (1 + self.transactionCosts)))
            self.holdings[t + 1] = - self.numberOfShares * self.data['Close'][t + 1]
            self.action[t + 1] = -1
            self.boughtAndHolding = False
            self.soldAndHolding = True

        # CASE 3: HOLD ACTION
        elif(action == 2 and (self.boughtAndHolding or self.soldAndHolding)):
            self.position[t + 1] = 0
            self.cash[t + 1] = self.cash[t]
            self.holdings[t + 1] = self.numberOfShares * self.data['Close'][t + 1]
            self.action[t + 1] = 2

        # CASE 4: Action chosen isn't possible in current state (bought and wants to buy again, sell before buying, etc.)
        else:
            # A possibility here is to give a negative reward in order to avoide missing a step due to an impossible action
            self.position[t + 1] = 0
            self.cash[t + 1] = self.cash[t]
            self.holdings[t + 1] = self.numberOfShares * self.data['Close'][t + 1]
            self.action[t + 1] = action



        # Update the total amount of money owned by the agent, as well as the return generated
        self.money[t + 1] = self.holdings[t + 1] + self.cash[t + 1]
        self.returns[t + 1] = (self.money[t + 1] - self.money[t])/self.money[t]

        # Set the RL reward returned to the trading agent
        self.reward = self.returns[t + 1]

        # Transition to the next trading time step
        self.t = self.t + 1
        self.state = [self.data['Open'][self.t - self.length : self.t].tolist(),
                      self.data['High'][self.t - self.length : self.t].tolist(),
                      self.data['Low'][self.t - self.length : self.t].tolist(),
                      self.data['Close'][self.t - self.length : self.t].tolist(),
                      self.data['Volume'][self.t - self.length : self.t].tolist(),
                      [self.position[self.t -1]]]
        if(self.t == self.data.shape[0] - 1):
            self.done = 1  

        self.info = None

        # Return the trading environment feedback to the RL trading agent
        return self.state, self.reward, self.done, self.info
    