import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arrow
from matplotlib.ticker import NullLocator
import gym
from gym import spaces

from action_emb.environments.utils import in_region


class GridworldContinuous(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, max_step_length = 0.2, max_steps = 30,
                 feature_dict=dict(obstacles=[{'x': 0, 'y': 0.25, 'x2': 0.5, 'y2': 0.3}],
                            goals=[{'coords': {'x': 0.25, 'y': 0.45, 'x2': 0.30, 'y2': 0.5}, 'reward': 1}])):
        print("Using continuous action space!")

        # Set up the action space
        self.action_space = spaces.Box(low=np.array([-1., -1.]), high=np.array([1., 1.]))

        # Set up the state space and make discrete if correct args passed
        self.observation_space = spaces.Box(low=np.zeros(2, dtype=np.float32),
                                            high=np.ones(2, dtype=np.float32),
                                            dtype=np.float32)

        self.feature_dict = feature_dict
        self.max_steps = max_steps

        # Set up the other environment params
        self.step_unit = 0.045
        self.repeat = int(max_step_length / self.step_unit)
        self.step_reward = -0.01
        self.collision_reward = -0.05

        self.first_render = True
        self.reset()

    def seed(self, seed):
        pass

    def reset(self):
        self.steps_taken = 0
        self.reward_states = self.get_reward_states()
        self.walls = self.get_static_obstacles()

        self.objects = {}
        self.curr_pos = np.array([.3, .1])
        self.curr_state = self.make_state()

        return self.curr_state

    def get_goal_rewards(self, pos):
        for i in range(len(self.reward_states)):
            region, reward = self.reward_states[i].values()
            if reward and in_region(pos, region):
                # This removes all captured reward states
                self.reward_states[i]['reward'] = 0
                print("Captured reward after steps:", self.steps_taken)
                return reward
        return 0

    def step(self, action):
        self.steps_taken += 1
        reward = 0
        terminal = self.is_terminal()

        if terminal:
            return self.curr_state, 0, terminal, {}
        # Look up the motion for the selected action and add the step reward
        motion = action
        reward += self.step_reward
        for i in range(self.repeat):
            delta = motion * self.step_unit
            new_pos = self.curr_pos + delta

            if self.valid_pos(new_pos):
                self.curr_pos = new_pos
                reward += self.get_goal_rewards(self.curr_pos)
            else:
                reward += self.collision_reward
                break

            if self.is_terminal():
                break
            self.curr_state = self.make_state()

        return self.curr_state.copy(), reward, self.is_terminal(), {}

    def is_terminal(self):
        for reward_state in self.reward_states:
            if in_region(self.curr_pos, reward_state['coords']):
                return True
        if self.steps_taken >= self.max_steps:
            return True
        else:
            return False

    def valid_pos(self, position):
        is_valid = True
        if not in_region(position, [0, 0, 1, 1]):
            is_valid = False

        for region in self.walls:
            if in_region(position, region):
                is_valid = False
                break

        return is_valid

    def make_state(self):
        x, y = self.curr_pos
        state = [x, y]
        return np.array(state)

    def get_static_obstacles(self):
        """
        Each obstacle is a solid bar, represented by (x,y,x2,y2)
        representing bottom left and top right corners,
        in percentage relative to board size

        :return: list of objects
        """
        obstacles = []
        obs_list = self.feature_dict['obstacles']
        for obs in obs_list:
            x, x2 = obs['x'], obs['x2']
            y, y2 = obs['y'], obs['y2']
            obstacles.append((x, y, x2, y2))
        return obstacles

    def get_reward_states(self):
        goals = []
        goal_list = self.feature_dict['goals']
        for goal in goal_list:
            x, x2 = goal['coords']['x'], goal['coords']['x2']
            y, y2 = goal['coords']['y'], goal['coords']['y2']
            goals.append({'coords': (x, y, x2, y2), 'reward': goal['reward']})
        return goals

    def render(self):
        x, y = self.curr_pos

        if self.first_render:
            self.first_render = False
            plt.figure(1, frameon=False)
            self.ax = plt.gca()
            plt.ion()

            for coords in self.walls:
                x1, y1, x2, y2 = coords
                w, h = x2 - x1, y2 - y1
                self.ax.add_patch(Rectangle((x1, y1), w, h, fill=True, color='gray'))

        # Add the reward states on the plot
        for reward_state in self.reward_states:
            coords, reward = reward_state.values()
            if reward:
                x1, y1, x2, y2 = coords
                w, h = x2 - x1, y2 - y1
                self.objects[coords] = Rectangle((x1, y1), w, h, fill=True, color='blue')
                self.ax.add_patch(self.objects[coords])

        # Add the agent on the plot
        self.objects['circle'] = Circle((x, y), 0.01, color='red')
        self.ax.add_patch(self.objects['circle'])
        plt.pause(1e-7)
        for _, item in self.objects.items():
            item.remove()
        self.objects = {}