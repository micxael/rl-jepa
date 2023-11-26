import torch
import torch.nn as nn
from torch.distributions import Normal

from ..algos.ppo.ppo_base import PGMLP


class GaussianMLP(nn.Module):
    def __init__(self, inp_dim, out_dim, hidden_sizes):
        super().__init__()
        self.net = PGMLP(
            [inp_dim, *hidden_sizes], activation=nn.ReLU, output_activation=nn.ReLU
        )

        self.mu_layer = nn.Linear(hidden_sizes[-1], out_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], out_dim)

    def forward(self, obs, deterministic=False):
        net_out = self.net(obs)

        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        std = torch.exp(log_std)

        out = mu if deterministic else Normal(mu, std).rsample()
        return out
