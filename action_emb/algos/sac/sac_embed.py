import itertools

import torch
import torch.nn as nn

import action_emb.embedding_modules as embedding_modules
from .sac_base import SquashedGaussianMLPActor, MLPQFunction, BaseMLPActorCritic
from ..exploration_strategies import Random
from ...utils.logger import EpochLogger

def mask_observation(obs_tensor, mask_percent=0.2):
    flattened_obs = obs_tensor.view(-1)
    num_elements_to_mask = int(flattened_obs.numel() * mask_percent)
    mask = torch.randperm(flattened_obs.numel())[:num_elements_to_mask]
    masked_obs = flattened_obs.clone()
    masked_obs[mask] = 0
    return masked_obs.view_as(obs_tensor)

class SACActorCriticEmbed(BaseMLPActorCritic):
    def __init__(self, observation_space, action_space, cnf):
        super(SACActorCriticEmbed, self).__init__(observation_space, action_space, cnf)

        # Embedding dimensionality for states and actions
        if cnf["embed"]["s_embed_dim"] is False:
            self.s_embed_dim = self.obs_dim
        else:
            self.s_embed_dim = cnf["embed"]["s_embed_dim"]
        self.a_embed_dim = cnf["embed"]["a_embed_dim"]
        self.cnf_embed = cnf["embed"]

        self.act_limit = self.cnf["training"]["act_limit"]

        self.pi = SquashedGaussianMLPActor(
            self.s_embed_dim,
            self.a_embed_dim,
            self.cnf_model["config"]["hidden_sizes_actor"],
            nn.Tanh,
            self.act_limit,
            self.cnf_model["config"]["std_dev_actor"],
        )

        # Set up the actor and q function (always continuous)
        self.q1 = MLPQFunction(
            self.s_embed_dim,
            self.a_embed_dim,
            self.cnf_model["config"]["hidden_sizes_critic"],
            nn.ReLU,
        )
        self.q2 = MLPQFunction(
            self.s_embed_dim,
            self.a_embed_dim,
            self.cnf_model["config"]["hidden_sizes_critic"],
            nn.ReLU,
        )

        self.q1_targ = MLPQFunction(
            self.s_embed_dim,
            self.a_embed_dim,
            self.cnf_model["config"]["hidden_sizes_critic"],
            nn.ReLU,
        )
        self.q2_targ = MLPQFunction(
            self.s_embed_dim,
            self.a_embed_dim,
            self.cnf_model["config"]["hidden_sizes_critic"],
            nn.ReLU,
        )

        self.q1_targ.load_state_dict(self.q1.state_dict())
        self.q2_targ.load_state_dict(self.q2.state_dict())

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.q1_targ.parameters():
            p.requires_grad = False
        for p in self.q2_targ.parameters():
            p.requires_grad = False

        self.setup_optims()

        # Setup the embedding module
        embedder_func = getattr(embedding_modules, cnf["embed"]["embed_module"])
        self.embedder = embedder_func(
            observation_space, action_space, self.act_dim, cnf
        )
        self.embedder_pretrain = False

        # Setup the exploration strategy for the embedder (in actual action space) and for ddpg (in embedding space)
        self.pretrain_strategy = Random(cnf, action_space)
        self.exploration_strategy = Random(cnf, None, embedded=True)

    def setup_optims(self):
        self.optimiser_pi = torch.optim.Adam(
            self.pi.parameters(), lr=self.cnf_train["learning_rate_pi"]
        )
        q_params = itertools.chain(self.q1.parameters(), self.q2.parameters())
        self.optimiser_q = torch.optim.Adam(
            q_params, lr=self.cnf_train["learning_rate_q"]
        )

    def step(self, state, episode_num, buffer, logger: EpochLogger):
        # For pre-training, we collect samples following a random policy
        if len(state.shape) < 2:
            state = state.unsqueeze(0)

        # Pick random actions in the actual action space
        if self.embedder_pretrain:
            action_info = {"num_actions": self.act_dim}
            action = self.pretrain_strategy.pick_action(action_info)
            return action

        # Pick random actions (points) in embedding space
        elif self.steps_taken < self.cnf_train["random_exp_steps"]:
            action_info = {"num_actions": self.a_embed_dim}
            action_embed = self.exploration_strategy.pick_action(action_info)
            with torch.no_grad():
                action = self.embedder.map_to_action(
                    torch.as_tensor(action_embed, dtype=torch.float32)
                )
            self.steps_taken += 1
            if isinstance(action, torch.Tensor):
                action = action.numpy()
            return action, action_embed

        with torch.no_grad():
            s_embed = self.embedder.get_state_embedding(state)
            action_embed, _ = self.pi(s_embed, False, False)
            action = self.embedder.map_to_action(action_embed)

            self.steps_taken += 1
        if (
            self.cnf_embed["continuous_learning"]
            and self.steps_taken % self.cnf_embed["update_every_n_steps"] == 0
        ):
            self.embedder.update(buffer, logger, False)
        # For the Mujoco environments, we have a tensor as the action while the other envs are already numpy
        if isinstance(action, torch.Tensor):
            action = action.numpy()
        return action, action_embed.numpy()

    def test_step(self, state):
        with torch.no_grad():
            s_embed = self.embedder.get_state_embedding(state, deterministic=True)
            action_embed, _ = self.pi(s_embed, True, False)
            action = self.embedder.map_to_action(action_embed)
        # For the Mujoco environments, we have a tensor as the action while the other envs are already numpy
        if isinstance(action, torch.Tensor):
            action = action.numpy()
        return action, action_embed.numpy()

    # Overwrite the original loss function to use the embedding representations
    def compute_loss_pi(self, experiences_dict):
        o = experiences_dict["obs"]
        with torch.no_grad():
            o_emb = self.embedder.get_state_embedding(o)

        pi, logp_pi = self.pi(o_emb)
        q1_pi = self.q1(o_emb, pi)
        q2_pi = self.q2(o_emb, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    # Overwrite the original loss function to use the embedding representation
    def compute_loss_q(self, experiences_dict):
        o, a, r, o2, d, a_emb = (
            experiences_dict["obs"],
            experiences_dict["act"],
            experiences_dict["rew"],
            experiences_dict["next_obs"],
            experiences_dict["done"],
            experiences_dict["act_emb"],
        )

        with torch.no_grad():
            o_emb = self.embedder.get_state_embedding(o)
            new_o2_sequence = torch.cat((o[:, -4 * o2.shape[1] :], mask_observation(o2)), dim=1)
            o2_emb = self.embedder.get_state_embedding(new_o2_sequence)

        q1 = self.q1(o_emb, a_emb)
        q2 = self.q2(o_emb, a_emb)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.pi(o2_emb)
            # Target Q-values
            q1_pi_targ = self.q1_targ(o2_emb, a2)
            q2_pi_targ = self.q2_targ(o2_emb, a2)
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

    def pretrain_embedder_update(self, buffer, logger):
        self.embedder.update(buffer, logger, pre_train=True)

    def load_embeddings(self):
        self.embedder.load_embeddings()

    def save_embeddings(self, logger, append=None):
        self.embedder.embedder.update_embeddings()
        self.embedder.save_embeddings(logger, append)

    def init_weights(
        self, pi, q, transition_weight=None, act_weight=None, state_weight=None
    ):
        self.pi = pi
        self.q = q
        self.embedder.init_weights(transition_weight, act_weight, state_weight)
        self.setup_optims()
