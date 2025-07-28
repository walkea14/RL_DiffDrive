# train_goal_nav.py
import numpy as np
import torch
from envs.goal_nav_2d import GoalNav2DEnv
from agents.ddpg_her import DDPGAgentHER, DEVICE

def flatten_obs(obs_dict):
    return obs_dict["observation"]

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
        ep_obs = env.reset()[0]
        agent.replay.start_episode()

        for t in range(steps_per_episode):
            o = flatten_obs(ep_obs)
            g = ep_obs["desired_goal"]
            a = agent.act(o, g, noise_scale=0.2)

            ep_obs_next, r, done, trunc, info = env.step(a)

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
    print("Training done.")
