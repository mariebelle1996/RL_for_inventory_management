# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 18:33:14 2020

@author: alami
"""


import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import time

import tqdm

from  RL_Agents.dqn_agent import Agent





def dqn(n_episodes, env, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """

    agent = Agent(state_size= len(env.observation_space), action_size=env.action_space.n, seed=0)

    action_list = np.array([])
    backlogs = []
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start   # initialize epsilon

    for i_episode in tqdm.tqdm(range(1, n_episodes+1)):
        state = env.reset()
        score = 0
        backlog = 0
        done = False
        while not done:
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            backlog += env.backlog
            if i_episode == n_episodes:
                action_list = np.append(action_list, action)
                #backlogs = np.append(backlogs, env.backlog)
                #state_list = np.append(state_list, env.state[0])
                

        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        backlogs.append(backlog)
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        #print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        #if i_episode % 100 == 0:
            #print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            
    return scores, action_list, backlogs




