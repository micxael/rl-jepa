import numpy as np
import torch
import torch.nn as nn

from gym.spaces import Box, Discrete
from sklearn.metrics.pairwise import euclidean_distances
from torch.utils import data as du

from action_emb.environments.utils import Space
from action_emb.utils.logger import EpochLogger
from ..nn import GaussianMLP
from ..SA_embedding_module_base import (
    SAEmbeddingModule,
    ContinuousMappingFunc,
    DiscreteMappingFunc,
)


class SASEmbeddingModule(SAEmbeddingModule):
    def __init__(self, observation_space, action_space, act_dim, cnf):
        super(SASEmbeddingModule, self).__init__(act_dim, cnf)
        self.embedder = SASEmbedder(
            observation_space,
            action_space,
            cnf["embed"]["s_embed_dim"],
            cnf["embed"]["a_embed_dim"],
            cnf["env"]["state_space_type"],
            cnf["env"]["action_space_type"],
        )
        self.optim = torch.optim.Adam(
            self.embedder.parameters(),
            lr=cnf["embed"]["learning_rate"],
            weight_decay=1e-4,
        )

        # This is used for continuous vs. discrete space embeddings
        self.action_space_type = cnf["env"]["action_space_type"]
        self.state_space_type = cnf["env"]["state_space_type"]
        self.cnf = cnf

        # Different loss functions depending on the state space type
        if self.state_space_type == "discrete":
            self.loss_fn = nn.NLLLoss()
            self.mapping_func = None
        else:
            self.loss_fn = nn.MSELoss()
        # This sets up the function g() to map the embedding point to an executable action
        if self.action_space_type == "continuous":
            self.mapping_func = ContinuousMappingFunc(
                cnf["embed"]["a_embed_dim"], action_space
            )
            self.mapping_func_eval = self.mapping_func
            self.mapping_loss = nn.MSELoss()
            self.mapping_optim = torch.optim.Adam(
                self.mapping_func.parameters(), lr=cnf["embed"]["learning_rate"]
            )
        elif self.action_space_type == "discrete":
            self.mapping_func = DiscreteMappingFunc(
                cnf["embed"]["a_embed_dim"], action_space
            )
            self.mapping_func_eval = self.mapping_func.get_action
            self.mapping_loss = nn.CrossEntropyLoss()
            self.mapping_optim = torch.optim.Adam(
                self.mapping_func.parameters(), lr=cnf["embed"]["learning_rate"]
            )
        elif self.action_space_type == "discreteNN":
            self.mapping_func = self.nearest_neighbour_action
            self.mapping_func_eval = self.mapping_func
        else:
            raise ValueError("Action space type for mapping func unknown.")

    def loss(self, pred, y):
        loss = self.loss_fn(pred, y)
        return loss

    def update(self, buffer, logger: EpochLogger, pre_train=False):
        if self.cnf["embed"]["use_full_buffer"] or pre_train:
            experiences_dict = buffer.get_data()
        elif self.cnf["embed"]["train_on_n_samples"] is not None:
            if buffer.enough_samples(self.cnf["embed"]["train_on_n_samples"]):
                experiences_dict = buffer.sample(
                    self.cnf["embed"]["train_on_n_samples"]
                )
            else:
                experiences_dict = buffer.sample(self.cnf["embed"]["batch_size"])
        else:
            raise ValueError(
                "Sampling instructions for embedding training missing!"
            )
        obs = experiences_dict["obs"]
        act = experiences_dict["act"]
        next_obs = experiences_dict["next_obs"]

        if self.state_space_type == "discrete":
            next_obs = torch.argmax(next_obs, -1).long()

        if self.action_space_type == "discreteNN":
            act_prep = self.prep_sas_actions(act, self.act_dim)
        else:
            act_prep = act
        data = SASDataset(obs, act_prep, next_obs, act)

        if pre_train:
            batch_size = self.cnf["embed"]["pre_train"]["batch_size"]
            train_iters = self.cnf["embed"]["pre_train"]["train_iters"]
        else:
            batch_size = self.cnf["embed"]["batch_size"]
            train_iters = self.cnf["embed"]["train_iters"]
        train_dataloader = du.DataLoader(data, batch_size=batch_size)
        self.train(train_iters, train_dataloader, pre_train, logger)

        # This stores the updated weight tables in a separate member
        self.embedder.update_embeddings()

    def train(self, train_iters, train_dataloader, pre_train, logger: EpochLogger):
        for _ in range(train_iters):
            self.embedder.train()
            for _, (s_x, a_x, y, a_raw) in enumerate(train_dataloader):

                self.optim.zero_grad()
                pred = self.embedder(s_x, a_x)
                loss = self.loss(pred, y)
                loss.backward()
                self.optim.step()
                if (
                    self.action_space_type == "continuous"
                    or self.action_space_type == "discrete"
                ):
                    self.mapping_optim.zero_grad()
                    act_emb = self.embedder.get_action_embedding(a_raw).detach()
                    act = self.mapping_func(act_emb).squeeze()
                    if self.action_space_type == "discrete":
                        act_prep = a_raw.long()
                    else:
                        act_prep = a_x
                    loss_mapping = self.mapping_loss(act, act_prep)
                    loss_mapping.backward()
                    self.mapping_optim.step()
                self.embedder.update_embeddings()
            if pre_train:
                logger.store(PreTrainLoss=loss.item(), overwrite=True, evaluate=True)
        logger.store(EmbeddingLoss=loss.item(), overwrite=True, evaluate=True)

    @staticmethod
    def prep_sas_actions(actions, act_dim):
        one_hot_acts = np.zeros((actions.shape[0], act_dim), dtype=np.float32)
        idx_1 = torch.from_numpy(np.arange(actions.shape[0]))
        actions = actions.long()
        one_hot_acts[idx_1, actions] = 1
        return one_hot_acts

    def nearest_neighbour_action(self, a_embed):
        all_actions = self.get_all_actions()
        dists = euclidean_distances(all_actions, a_embed.unsqueeze(0))
        act = np.argmin(dists)
        return act

    def map_to_action(self, a_emb):
        # Use mapping function for continuous or discrete. NN.shape available for discrete action spaces
        with torch.no_grad():
            return self.mapping_func_eval(a_emb)

    def get_action_from_emb(self, a_emb):
        # Note this is only used in the multi-armed Q network (also shouldn't work for continuous actions)
        return torch.matmul(a_emb, self.embedder.get_act_matrix().T)

    def init_weights(self, transition_weight=None, act_weight=None, state_weight=None):
        self.embedder.init_weights(transition_weight, act_weight, state_weight)
        self.optim = torch.optim.Adam(
            self.embedder.parameters(), lr=self.cnf["embed"]["learning_rate"]
        )


class SASEmbedder(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        s_embed_dim,
        a_embed_dim,
        state_space_type,
        action_space_type,
        sequence_length=5
    ):
        super(SASEmbedder, self).__init__()

        self.state_space_type = state_space_type
        self.action_space_type = action_space_type

        self.obs_dim = observation_space.shape[0]

        if isinstance(action_space, Box):
            self.act_dim = action_space.shape[0]
        elif isinstance(action_space, Discrete):
            self.act_dim = action_space.n
        elif isinstance(action_space, Space):
            self.act_dim = action_space.shape[0]

        # Set up the embedding layers
        self.sequence_length = sequence_length
        seq_obs_dim = self.obs_dim * self.sequence_length  # Adjusted for sequence length
        self.linear_state = GaussianMLP(seq_obs_dim, s_embed_dim, [s_embed_dim, ] * 1)  # phi
        self.act_state = nn.Tanh()

        self.linear_act = nn.Linear(self.act_dim, a_embed_dim, bias=False)  # g
        self.act_action = nn.Tanh()

        # Set up the consecutive layers
        self.transition = nn.Linear(s_embed_dim + a_embed_dim, self.obs_dim)  # P

        # LogSoftmax works with NLL Loss directly (unlike Softmax)
        if self.state_space_type == "discrete":
            self.activation = nn.LogSoftmax(dim=-1)
        else:
            self.activation = nn.Identity()

        self.update_embeddings()

    def forward(self, state_x, act_x):
        state_emb = self.act_state(self.linear_state(state_x, deterministic=True))
        act_emb = self.act_action(self.linear_act(act_x))
        x = torch.cat([state_emb, act_emb], dim=-1)
        x = self.activation(self.transition(x))
        return x

    def update_embeddings(self):
        # Note: This is only useful for indexing with discrete spaces
        self.state_embedding = self.act_state(self.linear_state.weight.data)
        self.act_embedding = torch.transpose(
            self.act_action(self.linear_act.weight.data), 0, 1
        )

    def get_act_matrix(self):
        # Note this should only be used with discrete embeddings
        return self.act_embedding

    def get_state_matrix(self):
        # Note this should only be used with discrete embeddings
        return self.state_embedding

    def get_state_embedding(self, state, deterministic=False):
        return self.act_state(self.linear_state(state, deterministic)).squeeze()

    def get_action_embedding(self, action):
        if (
            self.action_space_type == "discrete"
            or self.action_space_type == "discreteNN"
        ):
            return self.get_discrete_action_embedding(action)
        else:
            return self.get_continuous_action_embedding(action)

    def get_continuous_action_embedding(self, action):
        with torch.no_grad():
            return self.act_action(self.linear_act(action))

    def get_discrete_action_embedding(self, action):
        # Note: The casting to long might cause problems depending on the representation of actions
        return self.act_embedding[action.long()]

    def get_discrete_state_embedding(self, state):
        state = state != 0.0
        state_emb = self.state_embedding.transpose(1, 0)[None, :, :].repeat(
            state.shape[0], 1, 1
        )
        state_emb = state_emb[state, :]
        return state_emb.squeeze()

    def get_continuous_state_embedding(self, state):
        with torch.no_grad():
            return self.act_state(self.linear_state(state)).squeeze()

    def init_weights(self, transition_weight=None, act_weight=None, state_weight=None):
        if transition_weight is not None:
            self.transition.weight = nn.Parameter(transition_weight)
        if act_weight is not None:
            self.linear_act.weight = nn.Parameter(act_weight)
        if state_weight is not None:
            self.linear_state.weight = nn.Parameter(state_weight)
        self.update_embeddings()


class SASDataset(du.Dataset):
    def __init__(self, state_x, action_x, state_y, act_raw, batch_first=True):
        self.batch_first = batch_first
        self.state_x = state_x
        self.act_x = action_x
        self.y = state_y
        self.act_raw = act_raw

    def __getitem__(self, item):
        if self.batch_first:
            s_x = self.state_x[item]
            a_x = self.act_x[item]
            y = self.y[item]
            a_raw = self.act_raw[item]
            return s_x, a_x, y, a_raw
        else:
            s_x = self.state_x[:, item]
            a_x = self.act_x[:, item]
            y = self.y[:, item]
            a_raw = self.act_raw[:, item]
            return s_x, a_x, y, a_raw

    def __len__(self):
        if isinstance(self.state_x, list):
            return len(self.state_x)
        if self.batch_first:
            return self.state_x.shape[0]
        else:
            return self.state_x.shape[1]
