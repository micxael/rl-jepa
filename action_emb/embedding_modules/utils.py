import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as du


class TrajectoryData(du.Dataset):
    """
    Dataset object for Trajectory data to be used by DataLoader.
    """

    def __init__(self, train_x, train_y, batch_first=True):
        self.batch_first = batch_first
        self.X = train_x
        self.y = train_y

    def __getitem__(self, item):
        if self.batch_first:
            x_t = self.X[item]
            y_t = self.y[item]
            return x_t, y_t
        else:
            x_t = self.X[:, item]
            y_t = self.y[:, item]
            return x_t, y_t

    def __len__(self):
        if isinstance(self.X, list):
            return len(self.X)
        if self.batch_first:
            return self.X.shape[0]
        else:
            return self.X.shape[1]


class DoubleSidedTrajectoryData(du.Dataset):
    """
    Dataset object for two-sided trajectory data, i.e. the lhs trajectory, the rhs trajectory and
    the point in between.
    """

    def __init__(self, train_left, train_right, train_y, batch_first=True):
        self.batch_first = batch_first
        self.left = train_left
        self.right = train_right
        self.y = train_y

    def __getitem__(self, item):
        if self.batch_first:
            l_t = self.left[item]
            r_t = self.right[item]
            y_t = self.y[item]
            return l_t, r_t, y_t
        else:
            l_t = self.left[:, item]
            r_t = self.right[:, item]
            y_t = self.y[:, item]
            return l_t, r_t, y_t

    def __len__(self):
        if isinstance(self.left, list):
            return len(self.left)
        if self.batch_first:
            return self.left.shape[0]
        else:
            return self.left.shape[1]


def make_per_episode_data(episode, pred_len, full_pred=False):
    end_x = np.arange(pred_len, episode.shape[0] - pred_len + 1)
    start_x = 0
    # print(end_x)
    x = [episode[:i] for i in end_x]
    # print(x)

    if full_pred:
        end_idx = episode.shape[0]
        end_y = np.full(end_x.shape[0], end_idx, dtype=np.int)
    else:
        start_y = np.arange(pred_len - 1, episode.shape[0] - pred_len)
        end_y = start_y + pred_len + 1

    y = [episode[s:e] for s, e in zip(end_x, end_y)]
    return x, y


def convert_to_x_y(episodes, pred_len=2, full_pred=False):
    """
    Function to convert the episode data into a x and y set.
    :param episodes: List of the episode trajectories from the trajectory buffer.
    :param pred_len: Minimum prediction length, e.g. 2 means a minimum of 2 steps look-ahead.
    :param full_pred: If True, predict the entire remainder of the trajectory.
    :return:
    """
    eps = [make_per_episode_data(ep, pred_len, full_pred) for ep in episodes]
    x = [torch.from_numpy(part) for ep in eps for part in ep[0]]
    y = [torch.from_numpy(part) for ep in eps for part in ep[1]]
    return x, y


def make_per_episode_cgram_style_parts(episode):
    idx = np.arange(1, episode.shape[0] - 1)
    past = [episode[:i] for i in idx]
    future = [episode[i + 1:] for i in idx]
    y = [episode[i] for i in idx]
    return past, future, y


def convert_to_past_future_y(episodes):
    eps = [make_per_episode_cgram_style_parts(ep) for ep in episodes]
    past = [torch.from_numpy(part) for ep in eps for part in ep[0]]
    future = [torch.from_numpy(part) for ep in eps for part in ep[1]]
    y = [torch.from_numpy(part) for ep in eps for part in ep[2]]
    return past, future, y

def convert_to_x_y_complete_trajectory(obs, acts):
    obs = [torch.from_numpy(ep) for ep in obs]
    acts = [torch.from_numpy(ep) for ep in acts]
    return obs, acts

