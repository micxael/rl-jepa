import torch
import torch.nn as nn
from torch.distributions import Normal


class GaussianMLP(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()

        self.mu_layer = nn.Linear(inp_dim, out_dim)
        self.log_std_layer = nn.Linear(inp_dim, out_dim)

    def forward(self, obs, deterministic=False):
        mu = self.mu_layer(obs)
        log_std = self.log_std_layer(obs)
        std = torch.exp(log_std)

        out = mu if deterministic else Normal(mu, std).rsample()
        return out
