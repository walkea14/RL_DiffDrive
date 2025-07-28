# rl_path_planner/envs/grid_world.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class GridWorldEnv(gym.Env):
    """
    Simple 2D Grid World for RL path planning.
    Grid cells:
        0 = empty
        1 = obstacle
        2 = goal
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, grid_size=(7, 7), obstacles=None, start=(0, 0), goal=None):
        super(GridWorldEnv, self).__init__()

        self.grid_size = grid_size
        self.start = start
        self.goal = goal if goal else (grid_size[0] - 1, grid_size[1] - 1)

        # Define action space: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        self.action_space = spaces.Discrete(4)

        # Observation space: agent's (row, col) position
        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([grid_size[0] - 1, grid_size[1] - 1]),
            dtype=np.int32
        )

        # Initialize grid with obstacles
        self.grid = np.zeros(grid_size, dtype=int)
        if obstacles:
            for r, c in obstacles:
                self.grid[r, c] = 1
        self.grid[self.goal] = 2

        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.start
        return np.array(self.state, dtype=np.int32), {}

    def step(self, action):
        r, c = self.state
        if action == 0:   # UP
            r = max(r - 1, 0)
        elif action == 1: # RIGHT
            c = min(c + 1, self.grid_size[1] - 1)
        elif action == 2: # DOWN
            r = min(r + 1, self.grid_size[0] - 1)
        elif action == 3: # LEFT
            c = max(c - 1, 0)

        old_distance = np.linalg.norm(np.array(self.state) - np.array(self.goal))
        new_distance = np.linalg.norm(np.array([r,c]) - np.array(self.goal))

        # Check if obstacle
        if self.grid[r, c] == 1:
            reward = -5
            done = False
            # stay in place
            r, c = self.state
        elif (r, c) == self.goal:
            reward = 10
            done = True
        else:
            if new_distance < old_distance:
                reward = -0.5
            else:
                reward = -2
            #reward = -1
            done = False

        self.state = (r, c)
        return np.array(self.state, dtype=np.int32), reward, done, False, {}

    def render(self):
        grid_copy = self.grid.copy()
        r, c = self.state
        grid_copy[r, c] = 9  # mark agent
        print(grid_copy)

    def close(self):
        pass
