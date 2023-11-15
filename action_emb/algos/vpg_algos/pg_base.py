import numpy as np
from gym.spaces import Box, Discrete
from torch import nn as nn
import torch
from torch.distributions import Categorical, Normal

from ...environments.utils import Space
from ...utils.logger import EpochLogger


class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, action):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


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

        self.turn_off_exploration = False
        self.steps_taken = 0
        self.target_kl = self.cnf_train["target_kl"]

    def step(self, state, episode_num, buffer, logger: EpochLogger):
        raise ValueError("Need to implement this method in derived class.")

    def compute_loss_pi(self, obs, act, adv, logp_old):
        act = act.squeeze()
        pi, logp = self.pi(obs, act)
        loss_pi = -(logp * adv.squeeze()).mean()

        # Some extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent)

        return loss_pi, pi_info

    def compute_loss_v(self, obs, ret):
        pred = self.v(obs)
        loss = ((pred - ret) ** 2).mean()
        return loss

    def learn_policy(self, obs, act, adv, logp, a_embed=None):
        if a_embed is not None:
            loss_pi, pi_info = self.compute_loss_pi(obs, act, adv, logp, a_embed)
        else:
            loss_pi, pi_info = self.compute_loss_pi(obs, act, adv, logp)

        kl = pi_info["kl"]
        if kl > 1.5 * self.target_kl:
            return loss_pi.item()
        self.take_optimisation_step(self.optimiser_pi, self.pi, loss_pi)
        return loss_pi.item()

    def learn_critic(self, obs, ret):
        loss = self.compute_loss_v(obs, ret)
        self.take_optimisation_step(self.optimiser_v, self.v, loss)
        return loss.item()

    def take_optimisation_step(self, optim, network, loss):
        optim.zero_grad()
        loss.backward()
        optim.step()


def PGMLP(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class MLPCategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super(MLPCategoricalActor, self).__init__()
        self.logits_net = PGMLP([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, action):
        return pi.log_prob(action)


class MLPGaussianActor(Actor):
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
        activation,
        std_dev,
        output_activation=nn.Identity,
    ):
        super().__init__()
        log_std = std_dev * np.ones(act_dim, dtype=np.float32)
        self.std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = PGMLP(
            [obs_dim] + list(hidden_sizes) + [act_dim], activation, output_activation
        )

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = self.std
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(
            axis=-1
        )  # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super(MLPCritic, self).__init__()
        self.v_net = PGMLP([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)
