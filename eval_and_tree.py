# eval_and_tree.py
import torch
import numpy as np
from envs.goal_nav_2d import GoalNav2DEnv
from agents.ddpg_her import DDPGAgentHER, MLP, DEVICE
from utils.decision_tree import DecisionTreeLogger

def flatten_obs(obs_dict):
    return obs_dict["observation"]

if __name__ == "__main__":
    env = GoalNav2DEnv(world_size=5.0, success_radius=0.1)
    obs_dim = env.observation_space["observation"].shape[0]
    goal_dim = env.observation_space["desired_goal"].shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    # Build a "shell" agent to load weights into (only actor needed)
    actor = MLP(obs_dim + goal_dim, act_dim).to(DEVICE)
    actor.load_state_dict(torch.load("ddpg_goal_nav_actor.pt", map_location=DEVICE))
    actor.eval()

    def policy(obs, goal, noise=0.0):
        with torch.no_grad():
            inp = np.concatenate([obs, goal], axis=-1)
            a = actor(torch.as_tensor(inp, dtype=torch.float32, device=DEVICE)).cpu().numpy()
        if noise > 0:
            a += noise * np.random.randn(*a.shape)
        return np.clip(a, -act_limit, act_limit)

    # Log a branching exploration episode (epsilon/noise on)
    tree = DecisionTreeLogger()
    obs = env.reset()[0]
    o = flatten_obs(obs); g = obs["desired_goal"]

    root = tree.add_root(o)
    node = root
    noise = 0.2  # add some noise to see branching
    for _ in range(200):
        a = policy(o, g, noise=noise)
        obs_next, r, done, trunc, info = env.step(a)
        o2 = flatten_obs(obs_next)

        node = tree.add_transition(node, o2, a)

        o = o2
        g = obs_next["desired_goal"]
        if done or trunc:
            break

    tree.plot()
