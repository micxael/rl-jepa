import numpy as np
import torch
import torch.nn as nn

from gym.spaces import Box, Discrete
from torch.utils import data as du
from sklearn.metrics.pairwise import euclidean_distances
from action_emb.environments.utils import Space
from action_emb.utils.logger import EpochLogger
from ..SA_embedding_module_base import (
    SAEmbeddingModule,
    ContinuousMappingFunc,
    DiscreteMappingFunc,
)


class AutoEncoderEmbeddingModule(SAEmbeddingModule):
    def __init__(self, observation_space, action_space, act_dim, cnf):
        super(AutoEncoderEmbeddingModule, self).__init__(act_dim, cnf)
        self.embedder = AutoEncoderEmbedder(
            observation_space,
            action_space,
            cnf["embed"]["s_embed_dim"],
            cnf["embed"]["a_embed_dim"],
            cnf["env"]["state_space_type"],
            cnf["env"]["action_space_type"],
        )
        self.state_optim = torch.optim.Adam(
            self.embedder.state_embedder.parameters(),
            lr=cnf["embed"]["learning_rate"],
            weight_decay=1e-4,
        )

        self.action_optim = torch.optim.Adam(
            self.embedder.action_embedder.parameters(),
            lr=cnf["embed"]["learning_rate"],
            weight_decay=1e-4,
        )
        # This is used for continuous vs. discrete space embeddings
        self.action_space_type = cnf["env"]["action_space_type"]
        self.state_space_type = cnf["env"]["state_space_type"]
        self.cnf = cnf

        # Different loss functions depending on the state space type
        if self.state_space_type == "discrete":
            self.state_loss_fn = nn.NLLLoss()
            self.mapping_func = None
        else:
            self.state_loss_fn = nn.MSELoss()

        # Different loss functions depending on the state space type
        if (
            self.action_space_type == "discrete"
            or self.action_space_type == "discreteNN"
        ):
            self.action_loss_fn = nn.NLLLoss()
            self.mapping_func = None
        else:
            self.action_loss_fn = nn.MSELoss()

        # This sets up the function f() to map the embedding point to an executable action
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
            self.mapping_loss = nn.NLLLoss()
            self.mapping_optim = torch.optim.Adam(
                self.mapping_func.parameters(), lr=cnf["embed"]["learning_rate"]
            )
        elif self.action_space_type == "discreteNN":
            self.mapping_func = self.nearest_neighbour_action
            self.mapping_func_eval = self.mapping_func
        else:
            raise ValueError("Action space type for mapping func unknown.")

    def loss_state(self, pred, y):
        loss = self.state_loss_fn(pred, y)
        return loss

    def loss_action(self, pred, y):
        loss = self.action_loss_fn(pred, y)
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
        reconstruct_target = experiences_dict["obs"]
        if self.state_space_type == "discrete":
            reconstruct_target = torch.argmax(reconstruct_target, -1).long()

        if self.action_space_type == "discreteNN":
            act = act.long()
            act_prep = self.prep_sas_actions(act, self.act_dim)
        else:
            act_prep = act
        data = SSAADataset(obs, act_prep, reconstruct_target, act)

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
        for epoch in range(train_iters):
            self.embedder.train()
            for batch_idx, (s_x, a_x, s_y, a_raw) in enumerate(train_dataloader):
                self.action_optim.zero_grad()
                self.state_optim.zero_grad()
                state_pred, act_pred = self.embedder(s_x, a_x)
                loss_state = self.loss_state(state_pred, s_y)
                loss_action = self.loss_action(act_pred, a_raw)
                loss_state.backward()
                loss_action.backward()
                self.state_optim.step()
                self.action_optim.step()
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
                logger.store(
                    PreTrainLoss=loss_action.item(), overwrite=True, evaluate=True
                )
        logger.store(EmbeddingLoss=loss_action.item(), overwrite=True, evaluate=True)

    @staticmethod
    def prep_sas_actions(actions, act_dim):
        one_hot_acts = np.zeros((actions.shape[0], act_dim), dtype=np.float32)
        idx_1 = torch.from_numpy(np.arange(actions.shape[0]))
        t_actions = actions
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


class AutoEncoderEmbedder:
    def __init__(
        self,
        observation_space,
        action_space,
        s_embed_dim,
        a_embed_dim,
        state_space_type,
        action_space_type,
    ):
        self.state_space_type = state_space_type
        self.action_space_type = action_space_type

        self.obs_dim = observation_space.shape[0]

        if isinstance(action_space, Box):
            self.act_dim = action_space.shape[0]
        elif isinstance(action_space, Discrete):
            self.act_dim = action_space.n
        elif isinstance(action_space, Space):
            self.act_dim = action_space.shape[0]

        self.state_embedder = StateAutoEncoder(
            self.obs_dim, s_embed_dim, state_space_type
        )
        self.action_embedder = ActionAutoEncoder(
            self.act_dim, a_embed_dim, action_space_type
        )

        self.update_embeddings()

    def __call__(self, state_x, act_x):

        # Do the state embedding part
        state_y = self.state_embedder(state_x)
        # Do the action embedding part
        action_y = self.action_embedder(act_x)

        return state_y, action_y

    def train(self):
        self.state_embedder.train()
        self.action_embedder.train()

    @property
    def state_dict(self):
        return self.state_embedder.state_dict

    def update_embeddings(self):
        # Note: This is only useful for indexing with discrete spaces
        self.state_embedding = self.state_embedder.act_state(
            self.state_embedder.embed_layer_state.weight.data
        )
        self.act_embedding = torch.transpose(
            self.action_embedder.act_action(
                self.action_embedder.embed_layer_act.weight.data
            ),
            0,
            1,
        )

    def get_act_matrix(self):
        # Note this should only be used with discrete embeddings
        return self.act_embedding

    def get_state_matrix(self):
        # Note this should only be used with discrete embeddings
        return self.state_embedding

    def get_state_embedding(self, state):
        return self.state_embedder.act_state(
            self.state_embedder.embed_layer_state(state)
        ).squeeze()

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
            return self.action_embedder.act_action(
                self.action_embedder.embed_layer_act(action)
            )

    def get_discrete_action_embedding(self, action):
        # Note: The casting to long might cause problems depending on the representation of actions
        return self.act_embedding[action.long()]


class StateAutoEncoder(nn.Module):
    def __init__(self, obs_dim, s_embed_dim, state_space_type) -> None:
        super(StateAutoEncoder, self).__init__()
        self.obs_dim = obs_dim
        self.state_space_type = state_space_type

        # Set up the embedding layers for the state embedding
        self.embed_layer_state = nn.Linear(self.obs_dim, s_embed_dim, bias=False)
        self.act_state = nn.Tanh()
        self.reconstruct_layer_state = nn.Linear(s_embed_dim, self.obs_dim, bias=False)

        # LogSoftmax works with NLL Loss directly (unlike Softmax)
        if self.state_space_type == "discrete":
            self.state_activation = nn.LogSoftmax(dim=-1)
        else:
            self.state_activation = nn.Identity()

    def forward(self, state_x):
        state_emb = self.embed_layer_state(state_x)
        state_emb = self.act_state(state_emb)
        state_y = self.state_activation(self.reconstruct_layer_state(state_emb))
        return state_y


class ActionAutoEncoder(nn.Module):
    def __init__(self, act_dim, a_embed_dim, action_space_type) -> None:
        super(ActionAutoEncoder, self).__init__()
        self.act_dim = act_dim
        self.action_space_type = action_space_type

        # Set up the embeding layers for the action embedding
        self.embed_layer_act = nn.Linear(self.act_dim, a_embed_dim, bias=False)
        self.act_action = nn.Tanh()
        self.reconstruct_layer_act = nn.Linear(a_embed_dim, self.act_dim, bias=False)

        # LogSoftmax works with NLL Loss directly (unlike Softmax)
        if (
            self.action_space_type == "discrete"
            or self.action_space_type == "discreteNN"
        ):
            self.action_activation = nn.LogSoftmax(dim=-1)
        else:
            self.action_activation = nn.Identity()

    def forward(self, act_x):
        action_emb = self.embed_layer_act(act_x)
        action_emb = self.act_action(action_emb)
        action_y = self.action_activation(self.reconstruct_layer_act(action_emb))
        return action_y


class SSAADataset(du.Dataset):
    def __init__(
        self, state_x, action_x, reconstruct_target, act_raw, batch_first=True
    ):
        self.batch_first = batch_first
        self.state_x = state_x
        self.act_x = action_x
        self.act_raw = act_raw
        self.reconstruct_target = reconstruct_target

    def __getitem__(self, item):
        if self.batch_first:
            s_x = self.state_x[item]
            a_x = self.act_x[item]
            y = self.reconstruct_target[item]
            a_raw = self.act_raw[item]
            return s_x, a_x, y, a_raw
        else:
            s_x = self.state_x[:, item]
            a_x = self.act_x[:, item]
            y = self.reconstruct_target[:, item]
            a_raw = self.act_raw[:, item]
            return s_x, a_x, y, a_raw

    def __len__(self):
        if isinstance(self.state_x, list):
            return len(self.state_x)
        if self.batch_first:
            return self.state_x.shape[0]
        else:
            return self.state_x.shape[1]
