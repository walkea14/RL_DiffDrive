# rl_path_planner/main_train.py

import numpy as np
from envs.grid_world import GridWorldEnv
from agents.q_learning import QLearningAgent
from utils.visualize import GridVisualizer
import time

if __name__ == "__main__":
    obstacles = [(1,1), (2,2), (3,1)]
    env = GridWorldEnv(grid_size=(7, 7), obstacles=obstacles)
    agent = QLearningAgent(state_space=env.grid_size, action_space=env.action_space.n)

    visualizer = GridVisualizer(grid_size=env.grid_size, obstacles=obstacles, goal=env.goal)

    episodes = 300
    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            visualizer.handle_events()
            visualizer.set_agent_pos(state)
            visualizer.draw_grid()

            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            time.sleep(0.05)
            if done:
                break
        #if (ep+1) % 50 == 0:
            print(f"Episode {ep+1}/{episodes}, Total Reward: {total_reward}")

    # Save Q-table
    np.save("q_table.npy", agent.q_table)
    visualizer.close()
    print("Training complete. Q-table saved.")
