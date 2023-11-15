import numpy as np
import torch
from torch.distributions import Categorical, Normal

import os.path as osp, time, atexit, os
from gym.spaces import Box, Discrete
from torch import nn as nn
from torch.utils import data as du
import pickle

from action_emb.environments.utils import Space

from action_emb.utils.logger import EpochLogger


class SAEmbeddingModule:
    def __init__(self, act_dim, cnf):
        self.act_dim = act_dim
        self.cnf = cnf
        self.steps_taken = 0

    def step(self, buffer, logger: EpochLogger):
        self.steps_taken += 1

        if self.time_for_update() and buffer.enough_samples(
            self.cnf["embed"]["batch_size"]
        ):
            self.update(buffer, logger)
            return True
        return False

    def time_for_update(self):
        return self.steps_taken % self.cnf["embed"]["update_every_n_steps"] == 0

    def save_embeddings(self, logger, append=None):
        # Save the actual embedding tables
        if append == None:
            state_file = "state_embedding.pickle"
            action_file = "action_embedding.pickle"
        else:
            state_file = "state_embedding" + append + ".pickle"
            action_file = "action_embedding" + append + ".pickle"
        logger.save_file(self.embedder.state_embedding, state_file)
        logger.save_file(self.embedder.act_embedding, action_file)
        # Save the model state to load the embedder later
        logger.save_torch_model(self.embedder.state_dict(), "embedder.pt")

    def load_embeddings(self):
        print("------Loading Embeddings.....")
        folder = self.cnf["embed"]["load_emb_path"]
        state_file = "state_embedding.pickle"
        action_file = "action_embedding.pickle"
        with open(osp.join(folder, state_file), "rb") as out:
            self.embedder.state_embedding = pickle.load(out)
        with open(osp.join(folder, action_file), "rb") as out:
            self.embedder.act_embedding = pickle.load(out)

        self.embedder.load_state_dict(torch.load(osp.join(folder, "embedder.pt")))
        print("------Embedding successfully loaded--------")

    def get_all_actions(self):
        return self.embedder.get_act_matrix()

    def get_embedding_matrices(self):
        return self.embedder.get_state_matrix(), self.embedder.get_act_matrix()

    def get_action_embedding(self, action):
        return self.embedder.get_action_embedding(action)

    def get_state_embedding(self, state):
        return self.embedder.get_state_embedding(state)


class ContinuousMappingFunc(nn.Module):
    def __init__(self, a_embed_dim, action_space):
        super(ContinuousMappingFunc, self).__init__()
        # Get the action dimensionality
        if isinstance(action_space, Box):
            self.act_dim = action_space.shape[0]
        elif isinstance(action_space, Discrete):
            self.act_dim = action_space.n
        elif isinstance(action_space, Space):
            self.act_dim = action_space.shape[0]
        self.hidden_dim = 64
        self.linear_1 = nn.Linear(a_embed_dim, self.hidden_dim)
        self.act = nn.Sigmoid()
        self.linear_out = nn.Linear(self.hidden_dim, self.act_dim)

    def forward(self, emb):
        emb = self.act(self.linear_1(emb))
        return self.linear_out(emb)


class DiscreteMappingFunc(nn.Module):
    def __init__(self, a_embed_dim, action_space):
        super(DiscreteMappingFunc, self).__init__()
        if isinstance(action_space, Box):
            self.act_dim = action_space.shape[0]
        elif isinstance(action_space, Discrete):
            self.act_dim = action_space.n
        elif isinstance(action_space, Space):
            self.act_dim = action_space.shape[0]
        self.hidden_dim = 64

        self.linear_1 = nn.Linear(a_embed_dim, self.hidden_dim)
        self.act = nn.Tanh()
        self.linear_out = nn.Linear(self.hidden_dim, self.act_dim)
        self.act_out = nn.Softmax(dim=-1)

    def forward(self, emb):
        emb = self.act(self.linear_1(emb))
        return self.act_out(self.linear_out(emb))

    def get_action(self, emb):
        act_logits = self.forward(emb)
        dist = Categorical(logits=act_logits)
        act = dist.sample()
        return act
