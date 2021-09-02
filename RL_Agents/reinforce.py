# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 15:10:32 2020

@author: alami
"""


import gym
gym.logger.set_level(40) # suppress warnings (please remove if gives error)
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import torch
torch.manual_seed(0) # set random seed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import time



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self, env, h_size=2):
        super().__init__()
        s_size = len(env.observation_space)
        a_size = env.action_space.n

        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = np.array([state])
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)






gamma = 0.99

def reinforce(n_episodes, env, gamma):

	print_every = 100
	policy = Policy(env).to(device)
	optimizer = optim.Adam(policy.parameters(), lr=1e-2)
	scores = []
	scores_deque = deque(maxlen=100)
	for i_episode in range(1, n_episodes+1):
		saved_log_probs = []
		rewards = []
		state = env.reset()
		done = False
		while not done:
			action, log_prob = policy.act(state)
			saved_log_probs.append(log_prob)
			state, reward, done, _ = env.step(action)
			rewards.append(reward)

		scores_deque.append(sum(rewards))
		scores.append(sum(rewards))

		discounts = [gamma**i for i in range(len(rewards)+1)]
		R = sum([a*b for a,b in zip(discounts, rewards)])

		policy_loss = []
		for log_prob in saved_log_probs:
			policy_loss.append(-log_prob * R)
		policy_loss = torch.cat(policy_loss).sum()

		optimizer.zero_grad()
		policy_loss.backward()
		optimizer.step()

		if i_episode % print_every == 0:
			print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

	return scores




