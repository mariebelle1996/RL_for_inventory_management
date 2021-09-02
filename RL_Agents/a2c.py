# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 13:08:41 2020

@author: alami
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 13:03:25 2020

@author: alami
"""


import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Cart Pole

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=1, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()



SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self, env, h_size = 128):
        super(Policy, self).__init__()

        s_size = len(env.observation_space)
        a_size = env.action_space.n


        self.affine1 = nn.Linear(s_size, h_size)

        # actor's layer
        self.action_head = nn.Linear(h_size, a_size)

        # critic's layer
        self.value_head = nn.Linear(h_size, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values





def select_action(state, model):
	state = np.array([state])
	state = torch.from_numpy(state).float()
	probs, state_value = model(state)

    # create a categorical distribution over the list of probabilities of actions
	m = Categorical(probs)

    # and sample an action using the distribution
	action = m.sample()

    # save to action buffer
	model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # the action to take (left or right)
	return action.item()


def finish_episode(model, optimizer, eps):
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + args.gamma * R
        returns.insert(0, R)

    returns = torch.tensor(np.float_(returns))
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]


def A2C(env, n_episodes, print_every):


	model = Policy(env)
	optimizer = optim.Adam(model.parameters(), lr=3e-2)
	eps = np.finfo(np.float32).eps.item()
	scores = []


    # run inifinitely many episodes
	for i_episode in range(1,n_episodes+1):
        # reset environment and episode reward
		state = env.reset()
		rewards = []
        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning
		done = False
		while not done:

            # select action from policy
			action = select_action(state, model)

            # take the action
			state, reward, done, _ = env.step(action)

			rewards.append(reward)

			model.rewards.append(reward)

		scores.append(sum(rewards))


        # update cumulative rewar

        # perform backprop
		finish_episode(model, optimizer, eps)


		if i_episode % print_every == 0:
			print('Episode {}'.format(i_episode))

	return scores


