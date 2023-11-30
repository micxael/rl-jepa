import numpy as np
import torch
import torch.nn as nn
from ..ppo.ppo_base import PGMLP
from gym.spaces import Box, Discrete
from torch import nn as nn
import torch
import torch.nn.functional as F

from torch.distributions import Normal
from ...environments.utils import Space
from ...utils.logger import EpochLogger


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = PGMLP([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Ensure q has right shape.


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianMLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit, std_dev):
        super().__init__()
        self.net = PGMLP([obs_dim] + list(hidden_sizes), activation, activation)
        # Layers for mu and std dev, which share the initial few layers or use std
        # dev as an initializable parameter (comment out respectively)
        if std_dev is None:
            self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        else:
            log_std = std_dev * np.ones(act_dim, dtype=np.float32)
            self.std = torch.nn.Parameter(torch.as_tensor(log_std))

        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True, return_mu_log_var=False):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        if hasattr(self, "log_std_layer"):
            log_std = self.log_std_layer(net_out)
            log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(
                axis=1
            )
        else:
            logp_pi = None
        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        if return_mu_log_var:
            log_var = torch.log(std.pow(2))
            return pi_action, logp_pi, mu, log_var
        return pi_action, logp_pi


class BaseMLPActorCritic:
    def __init__(self, observation_space, action_space, cnf):

        if isinstance(observation_space, Discrete):
            self.obs_dim = observation_space.n
        else:
            self.obs_dim = observation_space.shape[0]
        self.action_space = action_space

        self.cnf_train = cnf["training"]
        self.cnf_model = cnf["model"]
        self.cnf = cnf
        if isinstance(action_space, Box):
            self.act_dim = action_space.shape[0]
        elif isinstance(action_space, Discrete):
            self.act_dim = action_space.n
        elif isinstance(action_space, Space):
            self.act_dim = action_space.shape[0]
        else:
            raise ValueError("Action Space Class is not known")

        self.polyak = self.cnf_train["polyak"]
        self.alpha = self.cnf_train["alpha"]
        self.turn_off_exploration = False
        self.steps_taken = 0

    def step(self, state, episode_num, buffer, logger: EpochLogger):
        raise ValueError("Need to implement this method in derived class.")

    def compute_loss_pi(self, experiences_dict):
        o = experiences_dict["obs"]
        pi, logp_pi = self.pi(o)
        q1_pi = self.q1(o, pi)
        q2_pi = self.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    def compute_loss_q(self, experiences_dict):
        o, a, r, o2, d = (
            experiences_dict["obs"],
            experiences_dict["act"],
            experiences_dict["rew"],
            experiences_dict["next_obs"],
            experiences_dict["done"],
        )

        q1 = self.q1(o, a)
        q2 = self.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.pi(o2)

            # Target Q-values
            q1_pi_targ = self.q1_targ(o2, a2)
            q2_pi_targ = self.q2_targ(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.cnf_train["gamma"] * (1 - d) * (
                q_pi_targ - self.alpha * logp_a2
            )

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(), Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    def update(self, buffer, logger):

        for i in range(self.cnf_train["update_iters"]):
            experiences_dict = buffer.sample(self.cnf_train["batch_size"])
            # First run one gradient descent step for Q1 and Q2
            self.optimiser_q.zero_grad()
            loss_q, q_info = self.compute_loss_q(experiences_dict)
            loss_q.backward()
            self.optimiser_q.step()

            # Record things
            logger.store(LossQ=loss_q.item(), **q_info)

            # Freeze Q-networks so you don't waste computational effort
            # computing gradients for them during the policy learning step.
            for p in self.q1.parameters():
                p.requires_grad = False
            for p in self.q2.parameters():
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            self.optimiser_pi.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(experiences_dict)
            loss_pi.backward()
            self.optimiser_pi.step()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in self.q1.parameters():
                p.requires_grad = True
            for p in self.q2.parameters():
                p.requires_grad = True

            # Record things
            logger.store(LossPi=loss_pi.item(), **pi_info)

            # Finally, update target Q networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(self.q1.parameters(), self.q1_targ.parameters()):
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

            with torch.no_grad():
                for p, p_targ in zip(self.q2.parameters(), self.q2_targ.parameters()):
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)


class SACReplayBuffer:
    """
    A simple FIFO experience replay buffer for 
    """

    def __init__(self, obs_dim, act_dim, act_emb_dim, size, sequence_length):
        self.sequence_length = sequence_length
        flattened_seq_dim = sequence_length * obs_dim
        self.obs_buf = np.zeros(combined_shape(size, flattened_seq_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.act_emb_buf = np.zeros(combined_shape(size, act_emb_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, act_emb, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.act_emb_buf[self.ptr] = act_emb
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            next_obs=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            act_emb=self.act_emb_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

    def enough_samples(self, num_samples):
        return self.size >= num_samples
