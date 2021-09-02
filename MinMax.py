# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 21:38:11 2021

@author: Marie-Belle BADR
"""



import numpy as np
import random

class MinMax:

    def __init__(self, env, minimum, maximum, min_default_order, seed=42):
        self.env = env
        self.rewards = np.array([])
        self.min = minimum
        self.max = maximum
        self.default_order = min_default_order
        #self.list_states = []
        #self.list_demands = []
        #self.list_actions = []
        self.env.reset()

    def step(self):
        if self.env.state[0] > self.min:
            action = 0
        else : 
            action = self.default_order
        
        #self.list_actions.append(action)
        #self.list_states.append(self.env.state)
        
        obs, reward, done, _ = self.env.step(action)
        #self.list_demands.append(demand)
        self.rewards = np.append(self.rewards, reward)
        
        
        


"""
obs = envs.reset()

n_max = 400

min_max = MinMax(envs, minimum = 2, maximum = n_max, min_default_order = n_max)

# Iterate
tt = 0
while tt < 365:
    min_max.step()
    # Store estimate of Q*
    tt +=1
        
   """    
