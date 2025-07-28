# rl_path_planner/agents/q_learning.py

import numpy as np

class QLearningAgent:
    def __init__(self, state_space, action_space, lr=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_space = state_space
        self.action_space = action_space
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize Q-table (grid_height x grid_width x actions)
        self.q_table = np.zeros((*state_space, action_space))

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space)
        return np.argmax(self.q_table[state[0], state[1], :])

    def update(self, state, action, reward, next_state, done):
        best_next_action = np.argmax(self.q_table[next_state[0], next_state[1], :])
        target = reward + self.gamma * self.q_table[next_state[0], next_state[1], best_next_action] * (not done)
        self.q_table[state[0], state[1], action] += self.lr * (target - self.q_table[state[0], state[1], action])

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
