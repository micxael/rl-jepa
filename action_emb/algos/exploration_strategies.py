import random

from gym.spaces import Box
import numpy as np
import torch

class ExplorationStrategy:
    def __init__(self, cnf, action_space):
        self.cnf = cnf

    def perturb_action_for_exploration(self, action_info):
        raise ValueError("Must be implemented")

    def reset(self):
        raise ValueError("Must be implemented")


class Random(ExplorationStrategy):
    def __init__(self, cnf, action_space, embedded=False):
        super(Random, self).__init__(cnf, action_space)
        if isinstance(action_space, Box) or embedded:
            self.policy = "uniform"
        else:
            self.policy = "categorical"

    def pick_action(self, action_info):
        if self.policy == 'uniform':
            act_dim = action_info["num_actions"]
            action = np.random.uniform(-1, 1, act_dim)
        else:
            act_dim = action_info["num_actions"]
            action = np.random.randint(0, act_dim)

        return action

    def pick_eval_action(self, action_info):
        return self.pick_action(action_info)


class EpsilonGreedy(ExplorationStrategy):
    def __init__(self, cnf):
        super(EpsilonGreedy, self).__init__(cnf)
        self.exploration_off = False

    def perturb_action_for_exploration(self, action_info, dim=1):
        action_values = action_info["action_values"]
        turn_off_exploration = action_info["turn_off_exploration"]
        episode_number = action_info["episode_number"]
        if turn_off_exploration and not self.exploration_off:
            self.exploration_off = True
        epsilon = self.get_updated_epsilon(action_info)
        if random.random() > epsilon or turn_off_exploration:
            return torch.argmax(action_values, dim=dim).item()
        action = np.random.randint(0, action_values.shape[1])
        return action

    def pick_eval_action(self, action_info, dim=1):
        action_values = action_info["action_values"]
        return torch.argmax(action_values, dim=dim).item()

    def get_updated_epsilon(self, action_info, epsilon=1.0):
        episode_number = action_info["episode_number"]
        if self.cnf["epsilon_decay_rate"]:
            epsilon = epsilon * (self.cnf["epsilon_decay_rate"] ** episode_number)
        else:
            epsilon = epsilon / (1.0 + (episode_number / self.cnf["epsilon_decay_denominator"]))
        return epsilon

    def reset(self):
        pass
