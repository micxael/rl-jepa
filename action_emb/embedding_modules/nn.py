import torch
import torch.nn as nn
from torch.distributions import Normal


class GaussianMLP(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()

        self.mu = nn.Linear(inp_dim, out_dim)
        self.log_var = nn.Linear(inp_dim, out_dim)

    def forward(self, obs, deterministic=False, return_mu_log_var=False):
        mu = self.mu(obs)
        log_var = self.log_var(obs)
        std = torch.exp(0.5 * log_var)

        out = mu if deterministic else Normal(mu, std).rsample()

        if return_mu_log_var:
            return mu, log_var

        return out
