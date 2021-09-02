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
    
    def __init__(self, product_name, IL0, max_IL, ordering_cost, holding_cost, penalty, demand_param, lead_time_param, horizon):
        
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
        
        
        # Initialize the time-series parameters
        
        self.demand_param = demand_param   # Parameter of the demand time serie
        self.LT_param = lead_time_param    # Parameter of the lead time time serie
        
        self.M = max_IL   # -> Maximum inventory level
        
        self.reset()   # reset the state vector
        self.seed()
        
        self.horizon = horizon
        
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    
    def reset(self):
        
        # Reset the state vector [InventoryLevel, InventoryTransition, Demand, Order, OnOrder]
        
        self.IL = self.IL0
        self.IT = 0
        self.Demand = 0
        self.Order =  0
        self.OnOrder = 0
        self.state = self._get_observation()
        
        self.lead_time = 0
        
        self.backlog = 0
        self.round = 0
        
        self.OnOrderList = []
        
        return self._get_observation()
        
        
    # Assume lead time and demand following a poisson distribution     
        
    def _generate_leadTime(self):
        return nr.poisson(self.LT_param)   # generation of an observed lead time
    
    def _generate_demand(self):
        return nr.poisson(self.demand_param) # generation of an observed demand
    
    
    # Getting leadTime and demand from given time series
    
    def _get_leadTime(self, LT_ts, t):
        return LT_ts[t]
    
    def _get_demand(self, demand_ts, t):
        return demand_ts[t]
    
    
    
    def _OnOrder_update(self, lead_time, action):
        
        if len(self.OnOrderList) > lead_time :
            self.OnOrderList[lead_time] += action
           
        else:
            self.OnOrderList.extend([0]*(lead_time -len(self.OnOrderList)+1))
            self.OnOrderList[lead_time] += action
            
    
    
    
    # Get the current state vector: [InventoryLevel, InventoryTransition, Demand, Order, OnOrder] 
    def _get_observation(self):
        return (self.IL, self.IT, self.Demand, self.Order, self.OnOrder)
    
    
    def _calculate_cost(self):
        
        return self.Co*self.Order + self.Ch*self.IL + self.Cp*self.backlog
        
    
    
    def step(self, action):
        
        self.round += 1
        
        self.Order = action
        
        self.lead_time = self._generate_leadTime()
        
        self._OnOrder_update(self.lead_time, self.Order)
        
        
        self.IT = self.OnOrderList.pop(0)
        
        self.OnOrder = sum(self.OnOrderList)
        

        self.Demand = self._generate_demand()
        
        
        self.backlog = max(self.Demand - min(self.IL + self.IT, self.M), 0)
               
        self.IL = max(min(self.IL + self.IT, self.M) - self.Demand, 0)
        
        reward = -1*self._calculate_cost()
        
        done = self.round == self.horizon
        
        self.state = self._get_observation()
        
        return self.state, reward, done, {}
    


if __name__ == "__main__":
    
    product_name = "pump"
    IL0 = 20
    max_IL = 400 
    ordering_cost = 10 
    holding_cost = 2
    penalty = 50
    demand_param = 6 
    lead_time_param = 2 
    horizon = 100
    
    env = inventoryProductEnv(product_name, IL0, max_IL, ordering_cost, holding_cost, penalty, demand_param, lead_time_param, horizon)
    
    
    """
    print(env.state)
    
    for i in range(50):
     
        action = random.randint(1,20)
        print(env.step(action))
        print(env.lead_time)
    """

    n_episodes = 2000

    scores, _ = dqn(n_episodes, env)

            
        
    
    
    
    
    
        
        
        
        
        
    
    
    
    
