from collections import namedtuple
import numpy as np
import torch

from .buffer import VPGFlexBuffer


class VPGEmbedBuffer(VPGFlexBuffer):
    def __init__(self, obs_dim, act_dim, gamma=1, lam=1):
        super(VPGEmbedBuffer, self).__init__(obs_dim, act_dim, gamma, lam)

        self.experience = namedtuple("experience",
                                     field_names=["obs", "act", "rew", "next_obs", "done",
                                                  "logp", 'val', 's_embed', 'a_embed'])

        self.s_embed_buf = []
        self.a_embed_buf = []

    def store(self, obs, act, rew, next_obs, done, logp, val, s_embed, a_embed):
        experience = self.experience(obs, act, rew, next_obs, done, logp, val, s_embed, a_embed)
        self.path_memory.append(experience)

    def finish_path(self, last_val=0):
        s_embed = [e.s_embed for e in self.path_memory]
        a_embed = [e.a_embed for e in self.path_memory]
        self.s_embed_buf.append(s_embed)
        self.a_embed_buf.append(a_embed)
        super(VPGEmbedBuffer, self).finish_path(last_val)

    def get(self):
        data = super(VPGEmbedBuffer, self).get()
        s_embed = torch.from_numpy(np.concatenate(self.s_embed_buf, axis=0)).float()
        a_embed = torch.from_numpy(np.concatenate(self.a_embed_buf, axis=0)).float()

        self.s_embed_buf = []
        self.a_embed_buf = []

        data['s_embed'] = s_embed
        data['a_embed'] = a_embed
        return data
