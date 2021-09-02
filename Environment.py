# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 11:46:10 2021

@author: alami
"""


import gym
from gym import spaces
from gym.utils import seeding
import numpy.random as nr
import random
import matplotlib.pyplot as plt
from scipy.stats import poisson


from RL_Agents import *

from RL_Agents.reinforce import *
from RL_Agents.dqn import *
from RL_Agents.a2c import *

#Generate a Poisson distribution
def poisson_distribution(k, mu, loc = 0):
    y = poisson.pmf(k, mu, loc)
    #plt.plot(x, y)
    #plt.show()
    return y

# Generate a stochastic process following a Poisson distribution
def generate_timeserie(param, length):
    return nr.poisson(param, length)

class inventoryProductEnv(gym.Env):
    
    def __init__(self, product_name, IL0, max_IL, fixed_cost, ordering_cost, holding_cost, penalty, demand_ts, lead_time_ts, horizon):
        
        # Initialize the state space and the action space
        
        self.product_name = product_name
        self.IL0 = IL0
        
        self.action_space = spaces.Discrete(max_IL+1)   # -> x_t
        self.observation_space = spaces.Tuple((
            spaces.Discrete(max_IL+1),   # -> Inventory Level
            spaces.Discrete(max_IL+1),   # -> Inventory transition  
            spaces.Discrete(max_IL+1),   # -> Demand
            spaces.Discrete(max_IL+1),   # -> Order
            spaces.Discrete(max_IL+1)    # -> On order
            ))
        
        # Initialize the product parameters (ordering cost, holding cost and penalty)
        
        self.Co = ordering_cost   # Ordering cost
        self.Ch = holding_cost    # Holding cost
        self.Cp = penalty         # Penalty
        self.fixed_cost = fixed_cost
                
        # Initialize the time-series parameters
        
        self.demand_ts = demand_ts   
        self.LT_ts = lead_time_ts   
        
        self.max_IL = max_IL   # -> Maximum inventory level
        
        self.reset()   # reset the state vector
        self.seed()
        
        self.horizon = horizon
        
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    
    def reset(self):
        #reset the state vector : state = [inventory_level, inventory_transition, demand, ordering level, on-order]
        self.IL = self.IL0
        self.IT = 0
        self.Demand = 0
        self.Order = 0
        self.OnOrder = 0
        self.state = self._get_observation()
         
        #reset lead time
        self.lead_time = 0
         
        #reset variables used 
        self.backlog = 0
        self.OnOrderList = []
               
        #reset period
        self.round = 0
          
        return self._get_observation()
      
    #get actual lead time and demand from generated time series
    def _get_leadtime(self):
        return self.LT_ts[self.round]
      
    def _get_demand(self):
        return self.demand_ts[self.round]
      
    def _OnOrder_update(self, leadtime, action):
        if len(self.OnOrderList) > leadtime:
            self.OnOrderList[leadtime] += action
        else:
            self.OnOrderList.extend([0]*int(leadtime - len(self.OnOrderList) + 1))
            self.OnOrderList[leadtime] += action
              
    #get the current state vector
    def _get_observation(self):
        return (self.IL, self.IT, self.Demand, self.Order, self.OnOrder)
          
    def _calculate_cost(self): 
        return self.fixed_cost + self.Co*self.Order + self.Ch*self.IL + self.Cp*self.backlog
                
    def step(self, action):  
        self.round += 1        
        self.Order = action        
        self.lead_time = self._get_leadtime()        
        self._OnOrder_update(self.lead_time, self.Order)                
        self.IT = self.OnOrderList.pop(0)        
        self.OnOrder = sum(self.OnOrderList)
        self.Demand = self._get_demand()        
        self.backlog = max(self.Demand - min(self.IL + self.IT, self.max_IL), 0)               
        self.IL = max(min(self.IL + self.IT, self.max_IL) - self.Demand, 0)       
        reward = -1*self._calculate_cost()        
        done = self.round == self.horizon        
        self.state = self._get_observation()
        
        return self.state, reward, done, {}
    


    
    
    
    
    
        
        
        
        
        
    
    
    
    
