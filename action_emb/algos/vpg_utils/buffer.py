from collections import namedtuple, deque

import numpy as np
import torch
import random
import action_emb.algos.vpg_utils.utils


class VPGFlexBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=1, lam=1):
        self.path_memory = []
        self.experience = namedtuple("experience",
                                     field_names=["obs", "act", "rew", "next_obs", "done", "logp", 'val', 'act_emb'])

        self.gamma = gamma
        self.lam = lam

        self.buffer = deque(maxlen=size)
        self.step_data = namedtuple("step",
                                    field_names=["obs", "act", 'act_emb', "rew", "next_obs", 'val', "done", "logp",
                                                 'adv', 'ret'])

    def store(self, obs, act, rew, next_obs, done, logp, val, act_emb=None):
        """
        Store trajectory steps in the path memory to store within an episode
        """
        experience = self.experience(obs, act, rew, next_obs, done, logp, val, act_emb)
        self.path_memory.append(experience)

    def finish_path(self, last_val=0):
        obs = [e.obs for e in self.path_memory]
        act = [e.act for e in self.path_memory]
        act_emb = [e.act_emb for e in self.path_memory]
        next_obs = [e.next_obs for e in self.path_memory]
        rew = [e.rew for e in self.path_memory]
        vals = [e.val for e in self.path_memory]
        logp = [e.logp for e in self.path_memory]
        done = [e.done for e in self.path_memory]

        rew.append(last_val)
        rew = np.vstack(rew)
        vals.append(last_val)
        vals = np.vstack(vals)

        deltas = rew[:-1] + self.gamma * vals[1:] - vals[:-1]
        adv = action_emb.algos.vpg_utils.utils.discount_cumsum(deltas, self.gamma * self.lam)
        ret = action_emb.algos.vpg_utils.utils.discount_cumsum(rew, self.gamma)[:-1]
        for t in range(len(obs)):
            self.buffer.append(
                self.step_data(obs[t], act[t], act_emb[t], rew[t], next_obs[t], vals[t], done[t], logp[t],
                               adv[t], ret[t]))

        self.path_memory = []

    def get_data(self):
        obs = np.array([e.obs for e in self.buffer if e is not None]).squeeze()
        next_obs = np.array([e.next_obs for e in self.buffer if e is not None]).squeeze()
        act = np.array([e.act for e in self.buffer if e is not None]).squeeze()
        act_emb = np.array([e.act_emb for e in self.buffer if e is not None]).squeeze()
        adv = np.array([e.adv for e in self.buffer if e is not None]).squeeze()
        # Implement the advantage normalisation trick
        adv = (adv - np.mean(adv)) / np.std(adv)

        rew = np.array([e.rew for e in self.buffer if e is not None]).squeeze()
        ret = np.array([e.ret for e in self.buffer if e is not None]).squeeze()
        val = np.array([e.val for e in self.buffer if e is not None]).squeeze()
        logp = np.array([e.logp for e in self.buffer if e is not None]).squeeze()
        done = np.array([e.done for e in self.buffer if e is not None]).squeeze()

        data = dict(obs=obs, next_obs=next_obs, act=act, act_emb=act_emb, rew=rew, ret=ret,
                    val=val, adv=adv, logp=logp, done=done)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

    def get(self):
        data = self.get_data()
        return data

    def reset(self):
        self.buffer.clear()

    def sample(self, num_samples=None):
        data = self.get_data()

        if num_samples is None:
            idx = np.random.choice(data['obs'].shape[0], size=self.batch_size)
        else:
            idx = np.random.choice(data['obs'].shape[0], size=num_samples)

        obs = data['obs'][idx]
        next_obs = data['next_obs'][idx]
        act = data['act'][idx]
        act_emb = data['act_emb'][idx]
        rew = data['rew'][idx]
        ret = data['ret'][idx]
        val = data['val'][idx]
        adv = data['adv'][idx]
        # Implement the advantage normalisation trick
        adv = (adv - torch.mean(adv)) / torch.std(adv)

        logp = data['logp'][idx]
        done = data['done'][idx]

        data = dict(obs=obs, next_obs=next_obs, act=act, act_emb=act_emb, rew=rew, ret=ret,
                    val=val, adv=adv, logp=logp, done=done)
        return data

    def sample_episodes(self, num_episodes=None):
        if num_episodes is not None:
            idx = np.random.choice(len(self.obs_buf), size=num_episodes)
        else:
            idx = np.arange(0, len(self.obs_buf))

        obs = np.array(self.obs_buf)[idx]
        next_obs = np.array(self.next_obs_buf)[idx]
        act = np.array(self.act_buf)[idx]
        act_emb = np.array(self.act_emb_buf)[idx]
        rew = np.array(self.rew_buf)[idx]
        ret = np.array(self.ret_buf)[idx]
        val = np.array(self.val_buf)[idx]
        adv = np.array(self.adv_buf)[idx]
        # Implement the advantage normalisation trick
        adv = (adv - torch.mean(adv)) / torch.std(adv)

        logp = np.array(self.logp_buf)[idx]
        done = np.array(self.done_buf)[idx]

        data = dict(obs=obs, next_obs=next_obs, act=act, act_emb=act_emb, rew=rew, ret=ret,
                    val=val, adv=adv, logp=logp, done=done)
        return data

    def enough_samples(self, batch_size):
        return len(self.buffer) > batch_size

    def __len__(self):
        return len(self.buffer)
