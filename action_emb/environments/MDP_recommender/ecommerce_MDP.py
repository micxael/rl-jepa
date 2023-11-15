import gym
from gym import spaces
import numpy as np
import pickle
from numpy.random import multinomial
import torch
import random
import os
import h5py


class RecommenderMDP(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, tr_1_path, tr_2_path, max_steps, items_map_path, items_path, prices_path, alpha, one_hot=False):
        super(RecommenderMDP, self).__init__()

        self.max_steps = max_steps
        self.one_hot = one_hot

        # Load the two transition matrices
        self.tr_1, self.tr_1_map = self.load_tr(tr_1_path)
        self.tr_2, self.tr_2_map = self.load_tr(tr_2_path)

        self.items = self.load_pickle(items_path)
        self.num_actions = len(self.items)
        self.num_states = len(self.tr_2)

        self.action_space = spaces.Discrete(n=self.num_actions)
        if one_hot:
            self.observation_space = spaces.MultiBinary(n=self.tr_2.shape[0])
        else:
            self.observation_space = spaces.MultiBinary(n=2 * self.num_actions)

        self.items_map = self.load_pickle(items_map_path)
        self.action_map = self.create_action_map(self.items_map)

        self.prices = self.load_pickle(prices_path)
        self.alpha = alpha

        # Set up the env in its initial state
        self.current_step = 0
        self.done = False
        start_item = random.choice(self.items)
        second_item = random.choice(self.items)
        self.state = tuple([start_item, second_item])

    def load_pickle(self, path):
        with (open(path, "rb")) as openfile:
            return pickle.load(openfile)

    def load_tr(self, tr_path):
        tr_mat_path = os.path.join(tr_path, 'tr_matrix.h5')
        hf = h5py.File(tr_mat_path, 'r')
        tr = hf.get('data')
        tr = np.array(tr)
        hf.close()

        map_path = os.path.join(tr_path, 'state_idx.pkl')
        with (open(map_path, "rb")) as openfile:
            map =pickle.load(openfile)
        return tr, map

    def create_action_map(self, item_map):
        action_map = {}
        for k, v in item_map.items():
            action_map[v] = k
        return action_map

    def calc_tr_prob(self, state):
        tr_1_key = tuple([state[1]])
        tr_1_idx = self.tr_1_map[tr_1_key]  # Last item to get tr_1 probs
        tr_1_probs = self.tr_1[tr_1_idx]

        tr_2_idx = self.tr_2_map[state]  # Both items to get the tr_2 probs
        tr_2_probs = self.tr_2[tr_2_idx]

        complete_probs = 0.5 * tr_1_probs + 0.5 * tr_2_probs
        return complete_probs

    def scale_probs_by_act(self, act, complete_probs):
        # Increase the probability of the recommended item by alpha and scale down the remaining ones accordingly
        scale = np.ones(act.shape)
        scale[act == 1] = self.alpha
        adj_prob = complete_probs * scale
        adj_prob[act == 1] += 0.05
        adj_prob = adj_prob / np.sum(adj_prob)
        return adj_prob

    def _get_transition(self, act):
        act_arr = np.zeros(self.num_actions)
        act_arr[act] = 1
        probs = self.calc_tr_prob(self.state)
        probs = self.scale_probs_by_act(act_arr, probs)
        next_item = np.random.choice(self.num_actions, 1, p=probs)
        item = self.action_map[next_item[0]]
        return item

    def _encode_state(self, state):
        if self.one_hot:
            s_enc = np.zeros(self.num_states)
            s_idx = self.tr_2_map[state]
            s_enc[s_idx] = 1
        else:
            s_first = state[0]
            s_second = state[1]
            s_enc = np.zeros(2*self.num_actions)
            s_first_idx = self.items_map[s_first]
            s_second_idx = self.items_map[s_second] + self.num_actions
            s_enc[s_first_idx] = 1
            s_enc[s_second_idx] = 1
        return s_enc

    def step(self, action):
        next_item = self._get_transition(action)
        reward = self.prices[next_item]
        done = False
        if self.current_step >= self.max_steps:
            self.done = True
            done = self.done
        self.state = tuple([self.state[1], next_item])
        enc_state =self._encode_state(self.state)
        return enc_state.copy(), reward, done, {}

    def reset(self):
        self.current_step = 0
        start_item = random.choice(self.items)
        second_item = random.choice(self.items)
        self.state = tuple([start_item, second_item])
        enc_state = self._encode_state(self.state)
        return enc_state