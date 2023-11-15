#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools as it
import gym
from gym import spaces
import numpy as np


class SlotMachine(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, no_reel, reel_range, no_value_per_reel, reward_threshold, max_step_size, bins, max_steps):
        super(SlotMachine, self).__init__()
        self.no_reel = no_reel  # number of reels
        self.reel_range = reel_range  # the maximum value of a reel symbol
        self.no_value_per_reel = no_value_per_reel  # each value represents a reel symbol
        self.reward_threshold = reward_threshold  # payout is trigger if the number of same symbols reach this threshold
        self.max_step_size = max_step_size
        self.reel_body = np.array([[2, 3, 6, 7, 0, 4],
                                   [3, 4, 6, 2, 1, 7],
                                   [1, 6, 5, 3, 4, 7],
                                   [5, 1, 4, 2, 6, 0]])
        self.max_no_step = max_steps
        self.bins = bins
        print(self.reel_body)

        self.pointer = np.zeros(self.no_reel)
        self.state = self.reel_body[range(self.no_reel), np.zeros(self.no_reel, dtype=int)]
        self.current_step = 0

        # Set up the one-hot action encoding
        per_reel_moves = [list(np.arange(0, self.bins)) for i in range(self.no_reel)]
        self.moves = np.array(list(it.product(*per_reel_moves)))

        # Set up the one-hot state encoding
        all_states = np.array(list(it.product(*self.reel_body)))

        self.num_states = all_states.shape[0]

        # Action is up to one full turn of each reel in bin steps
        self.action_space = spaces.Discrete(n=self.moves.shape[0])
        # State is one hot encoding
        self.observation_space = spaces.MultiBinary(n=all_states.shape[0])

        self.ep_reward = 0
        print("Slotmachine ENV set up.")

    def calc_state(self, state):
        idx = 0
        one_hot_idx = 0
        reel_lookup = self.reel_body[::-1]
        for i in state[::-1]:
            factor = np.argwhere(reel_lookup[idx] == i)[0][0]
            one_hot_idx += factor * (self.no_value_per_reel ** idx)
            idx += 1
        one_hot = np.zeros(self.num_states)
        one_hot[one_hot_idx] = 1
        return one_hot

    def _calculate_reward(self, current_state):
        reward = 0

        # Count the unique signs showing
        unq_val, unq_count = np.unique(current_state, return_counts=True)
        for count, idx in zip(unq_count, range(len(unq_count))):
            # For all signs where more than reward_threshold are showing, multiply the no. with the sign
            if count >= self.reward_threshold:
                reward += count * unq_val[idx]
        return reward

    def _take_action(self, action):
        self.current_step = self.current_step + 1
        self.done = False
        move = self.moves[action]
        move = self.max_step_size / self.bins * move
        new_proto_pointer = self.pointer + np.floor(move * self.no_value_per_reel)
        self.new_pointer = np.array([new_proto_pointer[i] if new_proto_pointer[i] <= self.no_value_per_reel - 1 else
                                     new_proto_pointer[i] - self.no_value_per_reel for i in
                                     range(len(new_proto_pointer))])
        self.new_pointer = self.new_pointer.astype(int)
        self.state = self.reel_body[range(self.no_reel), self.new_pointer]
        self.pointer = self.new_pointer
        self.reward = self._calculate_reward(self.state)
        if self.current_step >= self.max_no_step:
            self.done = True

    def _next_observation(self):
        obs = self.calc_state(self.state)
        return obs

    def step(self, action):
        self._take_action(action)
        obs = self._next_observation()
        reward = self.reward
        done = self.done
        self.ep_reward += reward
        if done:
            self.ep_reward = 0
        return obs, reward, done, {}

    def reset(self):
        self.current_step = 0
        self.pointer = np.zeros(self.no_reel)
        self.state = self.reel_body[range(self.no_reel), np.zeros(self.no_reel, dtype=int)]
        return self._next_observation()
