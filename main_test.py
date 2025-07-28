# rl_path_planner/main_test.py

import numpy as np
import time
from envs.grid_world import GridWorldEnv
from utils.visualize import GridVisualizer

if __name__ == "__main__":
    obstacles = [(1,1), (2,2), (3,1)]
    env = GridWorldEnv(grid_size=(7, 7), obstacles=obstacles)
    q_table = np.load("q_table.npy")

    visualizer = GridVisualizer(grid_size=(7,7), obstacles=obstacles, goal=env.goal)

    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        visualizer.handle_events()
        visualizer.set_agent_pos(state)
        visualizer.draw_grid()

        action = np.argmax(q_table[state[0], state[1], :])
        state, reward, done, _, _ = env.step(action)
        total_reward += reward

        time.sleep(0.5)

        #env.render()
        print(f"Step reward: {reward}")

    print(f"Episode finished. Total reward: {total_reward}")
    time.sleep(2)
    visualizer.close()
