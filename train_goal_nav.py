# train_goal_nav.py

import numpy as np
import torch
from envs.goal_nav_2d import GoalNav2DEnv
from agents.ddpg_her import DDPGAgentHER, DEVICE
from utils.live_plot import LivePlot2D

multi_goal = False
fixed_goal = np.array([2.0, -2.0])  # pick any reachable goal


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
        collision_penalty=-10.0,
        terminate_on_collision=True
    )

    obs_dim = env.observation_space["observation"].shape[0]
    goal_dim = env.observation_space["desired_goal"].shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    agent = DDPGAgentHER(
        obs_dim=obs_dim,
        goal_dim=goal_dim,
        act_dim=act_dim,
        act_limit=act_limit,
        buffer_size=200_000
    )

    # LIVE PLOT
    viz = LivePlot2D(world_size=env.world_size, obstacles=env.obstacles)

    episodes = 1000
    steps_per_episode = env.max_steps

    for ep in range(episodes):
        ep_obs = env.reset()[0]
        agent.replay.start_episode()
        viz.reset_trail()

        for t in range(steps_per_episode):
            o = flatten_obs(ep_obs)
            g = ep_obs["desired_goal"]
            a = agent.act(o, g, noise_scale=0.2)
            ep_obs_next, r, done, trunc, info = env.step(a)

            agent.replay.store_step(
                obs=o,
                ag=ep_obs["achieved_goal"],
                g=g,
                act=a,
                rew=r,
                done=float(done),
                obs_next=flatten_obs(ep_obs_next),
                ag_next=ep_obs_next["achieved_goal"]
            )

            ep_obs = ep_obs_next

            # Visualize
            viz.update(ep_obs["observation"], g)
            viz.render()

            if done or trunc:
                break

        agent.replay.end_episode()
        for _ in range(40):
            agent.update()

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{episodes}")

    torch.save(agent.actor.state_dict(), "ddpg_goal_nav_actor.pt")
    print("Training done.")


""" # train_goal_nav.py
import numpy as np
import torch
from envs.goal_nav_2d import GoalNav2DEnv
from agents.ddpg_her import DDPGAgentHER, DEVICE
from utils.live_plot import LivePlot2D


def flatten_obs(obs_dict):
    return obs_dict["observation"]

obstacles = [
    (0.0, 0.0, 0.5),
    (1.5, -1.0, 0.3),
    (-1.0, 1.2, 0.4)
]

env = GoalNav2DEnv(
    world_size=5.0,
    success_radius=0.2,
    obstacles=obstacles,
    collision_penalty=-10.0,
    terminate_on_collision=True
)

viz = LivePlot2D(world_size=env.world_size, obstacles=env.obstacles)



if __name__ == "__main__":
    env = GoalNav2DEnv(world_size=5.0, success_radius=0.1)
    obs_dim = env.observation_space["observation"].shape[0]
    goal_dim = env.observation_space["desired_goal"].shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    agent = DDPGAgentHER(
        obs_dim=obs_dim,
        goal_dim=goal_dim,
        act_dim=act_dim,
        act_limit=act_limit,
        buffer_size=200000
    )

    episodes = 2000
    steps_per_episode = env.max_steps

    for ep in range(episodes):
        viz.reset_trail()

        ep_obs = env.reset()[0]
        agent.replay.start_episode()

        for t in range(steps_per_episode):
            o = flatten_obs(ep_obs)
            g = ep_obs["desired_goal"]
            a = agent.act(o, g, noise_scale=0.2)

            ep_obs_next, r, done, trunc, info = env.step(a)

            viz.update(agent.state, agent.goal)
            viz.render()


            # store transition
            agent.replay.store_step(
                obs=o,
                ag=ep_obs["achieved_goal"],
                g=g,
                act=a,
                rew=r,
                done=float(done),
                obs_next=flatten_obs(ep_obs_next),
                ag_next=ep_obs_next["achieved_goal"]
            )

            ep_obs = ep_obs_next
            if done or trunc:
                break

        agent.replay.end_episode()

        # Update for a few gradient steps
        for _ in range(40):
            agent.update()

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{episodes}")

    # Save actor
    torch.save(agent.actor.state_dict(), "ddpg_goal_nav_actor.pt")
    print("Training done.")"""
