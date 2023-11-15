import numpy as np

from action_emb.algos.shared_utils import create_dataloader
import torch
import torch.nn as nn
from action_emb.algos.vpg_algos.pg_base import (
    MLPGaussianActor,
    MLPCritic,
)
from .pg_base import BaseMLPActorCritic
from ...utils.logger import EpochLogger
import action_emb.embedding_modules as embedding_modules
from ..exploration_strategies import Random


class VPGActorCriticEmbed(BaseMLPActorCritic):
    def __init__(self, observation_space, action_space, cnf):
        super(VPGActorCriticEmbed, self).__init__(observation_space, action_space, cnf)

        # Embedding dimensionality for states and actions
        if cnf["embed"]["s_embed_dim"] is False:
            self.s_embed_dim = self.obs_dim
        else:
            self.s_embed_dim = cnf["embed"]["s_embed_dim"]
        self.a_embed_dim = cnf["embed"]["a_embed_dim"]
        self.cnf_embed = cnf["embed"]

        # Actor that takes an embedded state and outputs an embedded action with Gaussian dist.
        self.pi = MLPGaussianActor(
            self.s_embed_dim,
            self.a_embed_dim,
            self.cnf_model["config"]["hidden_sizes_actor"],
            nn.Tanh,
            self.cnf_model["config"]["std_dev_actor"],
        )

        # Critic also takes and embedded state and outputs a single value
        self.v = MLPCritic(
            self.s_embed_dim, self.cnf_model["config"]["hidden_sizes_critic"], nn.Tanh
        )

        self.setup_optims()

        # Setup the embedding module
        embedder_func = getattr(embedding_modules, cnf["embed"]["embed_module"])
        self.embedder = embedder_func(
            observation_space, action_space, self.act_dim, cnf
        )

        self.embedder_pretrain = False
        self.pretrain_strategy = Random(cnf, action_space)

    def setup_optims(self):
        self.optimiser_pi = torch.optim.Adam(
            self.pi.parameters(), lr=self.cnf_train["learning_rate_pi"], eps=1e-4
        )
        self.optimiser_v = torch.optim.Adam(
            self.v.parameters(), lr=self.cnf_train["learning_rate_v"], eps=1e-4
        )

    def step(self, state, episode_num, buffer, logger: EpochLogger):
        # For pre-training, we collect samples following a random policy
        if len(state.shape) < 2:
            state = state.unsqueeze(0)
        if self.embedder_pretrain:
            action_info = {"num_actions": self.act_dim}
            action = self.pretrain_strategy.pick_action(action_info)
            return action
        with torch.no_grad():
            s_embed = self.embedder.get_state_embedding(state)
            pi = self.pi._distribution(s_embed)
            action_embed = pi.sample()
            logprob_action = self.pi._log_prob_from_distribution(pi, action_embed)
            v = self.v(s_embed)
            action = self.embedder.map_to_action(action_embed)
            self.steps_taken += 1
        if (
            self.cnf_embed["continuous_learning"]
            and self.steps_taken % self.cnf_embed["update_every_n_steps"] == 0
        ):
            self.embedder.update(buffer, logger, False)

        # For the Mujoco environments, we have a tensor as the action whil the other envs are already numpy
        if isinstance(action, torch.Tensor):
            action = action.numpy()
        return action, v.numpy(), logprob_action.numpy(), action_embed.numpy()

    def update(self, buffer, logger):
        experiences_dict = buffer.get()
        dataloader = create_dataloader(experiences_dict, self.cnf_train["batch_size"])

        for i in range(
            max(self.cnf_train["num_iters_pi"], self.cnf_train["num_iters_v"])
        ):
            for batch_idx, (obs, act, adv, logp, ret, a_embed) in enumerate(dataloader):
                if i <= self.cnf_train["num_iters_pi"]:
                    pi_loss = self.learn_policy(obs, act, adv, logp, a_embed)
                if i <= self.cnf_train["num_iters_v"]:
                    v_loss = self.learn_critic(obs, ret)

        logger.store(LossPi=pi_loss, LossV=v_loss)

    # Overwrite the original loss function to use the embedding representations
    def compute_loss_pi(self, obs, act, adv, logp_old, a_embed):
        s_embed = self.embedder.get_state_embedding(obs)

        pi, logp = self.pi(s_embed, a_embed)

        loss_pi = -(logp * adv).mean()

        # Some extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent)

        return loss_pi, pi_info

    # Overwrite the original loss function to use the embedding representation
    def compute_loss_v(self, obs, ret):
        s_embed = self.embedder.get_state_embedding(obs)
        pred = self.v(s_embed)
        loss = ((pred - ret) ** 2).mean()
        return loss

    def pretrain_embedder_update(self, buffer, logger):
        self.embedder.update(buffer, logger, pre_train=True)

    def load_embeddings(self):
        self.embedder.load_embeddings()

    def save_embeddings(self, logger, append=None):
        self.embedder.embedder.update_embeddings()
        self.embedder.save_embeddings(logger, append)

    def init_weights(
        self, pi, v, transition_weight=None, act_weight=None, state_weight=None
    ):
        self.pi = pi
        self.v = v
        self.embedder.init_weights(transition_weight, act_weight, state_weight)
        self.setup_optims()