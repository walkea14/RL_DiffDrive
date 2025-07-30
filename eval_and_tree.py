# eval_and_tree.py

import torch
import numpy as np
from envs.goal_nav_2d import GoalNav2DEnv
from agents.ddpg_her import MLP, DEVICE
from utils.live_plot import LivePlot2D

def flatten_obs(obs_dict):
    return obs_dict["observation"]

if __name__ == "__main__":
    obstacles = [
        (0.0, 0.0, 0.5),
        (1.5, -1.0, 0.3),
        (-1.0, 1.2, 0.4)
    ]

    env = GoalNav2DEnv(
        world_size=5.0,
        success_radius=0.2,
        obstacles=obstacles,
        terminate_on_collision=True
    )

    obs_dim = env.observation_space["observation"].shape[0]
    goal_dim = env.observation_space["desired_goal"].shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    actor = MLP(obs_dim + goal_dim, act_dim).to(DEVICE)
    actor.load_state_dict(torch.load("ddpg_goal_nav_actor.pt", map_location=DEVICE))
    actor.eval()

    def policy(obs, goal):
        with torch.no_grad():
            inp = np.concatenate([obs, goal], axis=-1)
            a = actor(torch.tensor(inp, dtype=torch.float32, device=DEVICE)).cpu().numpy()
        return np.clip(a, -act_limit, act_limit)

    viz = LivePlot2D(world_size=env.world_size, obstacles=obstacles)
    viz.reset_trail()

    obs = env.reset()[0]
    o = flatten_obs(obs)
    g = obs["desired_goal"]

    for _ in range(200):
        a = policy(o, g)
        obs_next, r, done, trunc, info = env.step(a)
        o = flatten_obs(obs_next)
        g = obs_next["desired_goal"]

        viz.update(o, g)
        viz.render()

        if done or trunc:
            break

    viz.save_gif("evaluation_trajectory.gif")
