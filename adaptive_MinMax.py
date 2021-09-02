# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 10:06:48 2021

@author: ELIE
"""
import numpy as np
from Environment import *

class adaptive_MinMax:
    def __init__(self, env, performance, s0, seed = 42):
        self.env = env
        self.performance = performance
        self.s = s0
        
        #reset environment
        self.env.reset()
        
        #create an array to store previous demands that serve to compute the order-level
        self.list_demands = np.array([])
        
        #create an array to store rewards in order to compute the total cost
        self.rewards = np.array([])
        self.action = 0
                
        #i variable allows to place an order only once when the inventory level goes under the order-level s
        self.i = 0
        
        #variables used to compute the ordering quantity Q
        self.fixed_cost = self.env.fixed_cost
        self.carrying_charge = self.env.Ch / self.env.Co
        
        #Compute the fixed part of Q 
        self.Q_fixed = (2*self.fixed_cost / self.env.Co*self.carrying_charge)**0.5
        
    def step(self):
        
        if self.env.state[0] > self.s:
            action = 0
            self.i = 0
        elif(self.i == 0):
            self.lead_time = self.env.lead_time
            #action = self.env.max_IL
            action = self.Q_fixed*(self.env.Demand**0.5)
            self.i = 1
        elif(self.i == 1):
            if self.lead_time >= 0:
                self.lead_time -= self.lead_time
                self.list_demands = np.append(self.list_demands, self.env.Demand)
                action = 0      
            else:
                self.list_demands = np.append(self.list_demands, self.env.Demand)
                n = len(self.list_demands) 
                self.list_demands = self.list_demands.sort()  
                k = (n+1)*self.performance
                y = int(k)
                w = k - y
                if k > n:
                    self.s = self.list_demands[n-1]
                else: 
                    self.s = (1-w)*self.list_demands[y-1] + w*self.list_demands[y]
                action = 0
                self.list_demands = np.array([]) 
                
        obs, reward, done, _ = self.env.step(action)
        self.rewards = np.append(self.rewards, reward)  
     
            
            
                
            