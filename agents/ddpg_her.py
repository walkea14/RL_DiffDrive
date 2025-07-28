# agents/ddpg_her.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_tensor(x):
    return torch.as_tensor(x, dtype=torch.float32, device=DEVICE)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256,256), act=nn.ReLU):
        super().__init__()
        layers = []
        dims = (in_dim,) + hidden
        for i in range(len(hidden)):
            layers += [nn.Linear(dims[i], dims[i+1]), act()]
        layers += [nn.Linear(hidden[-1], out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class DDPGAgentHER:
    def __init__(
        self,
        obs_dim,
        goal_dim,
        act_dim,
        act_limit,
        gamma=0.98,
        tau=0.005,
        lr=1e-3,
        her_ratio=0.8,
        buffer_size=1000000,
        batch_size=256,
    ):
        self.gamma = gamma
        self.tau = tau
        self.act_limit = act_limit
        self.batch_size = batch_size
        self.her_ratio = her_ratio

        # Actor: (obs+goal) -> action
        self.actor = MLP(obs_dim + goal_dim, act_dim).to(DEVICE)
        self.actor_targ = MLP(obs_dim + goal_dim, act_dim).to(DEVICE)
        self.actor_targ.load_state_dict(self.actor.state_dict())

        # Critic: (obs+goal, action) -> Q
        self.critic = MLP(obs_dim + goal_dim + act_dim, 1).to(DEVICE)
        self.critic_targ = MLP(obs_dim + goal_dim + act_dim, 1).to(DEVICE)
        self.critic_targ.load_state_dict(self.critic.state_dict())

        self.pi_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.replay = HERReplayBuffer(
            obs_dim, goal_dim, act_dim, size=buffer_size, her_ratio=her_ratio
        )

    @torch.no_grad()
    def act(self, obs, goal, noise_scale=0.1):
        inp = np.concatenate([obs, goal], axis=-1)
        a = self.actor(to_tensor(inp)).cpu().numpy()
        a += noise_scale * np.random.randn(*a.shape)
        return np.clip(a, -self.act_limit, self.act_limit)

    def update(self):
        if self.replay.size < self.batch_size:
            return

        batch = self.replay.sample_batch(self.batch_size)
        o, g, a, r, o2, g2, d = [to_tensor(x) for x in batch]

        # Target actions and Q
        with torch.no_grad():
            a2 = self.actor_targ(torch.cat([o2, g2], dim=-1))
            q_targ = self.critic_targ(torch.cat([o2, g2, a2], dim=-1))
            backup = r + self.gamma * (1 - d) * q_targ

        # Critic loss
        q = self.critic(torch.cat([o, g, a], dim=-1))
        q_loss = F.mse_loss(q, backup)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Actor loss (maximize Q)
        pi = self.actor(torch.cat([o, g], dim=-1))
        q_pi = self.critic(torch.cat([o, g, pi], dim=-1))
        pi_loss = -q_pi.mean()

        self.pi_optimizer.zero_grad()
        pi_loss.backward()
        self.pi_optimizer.step()

        # Polyak averaging
        with torch.no_grad():
            for p, p_targ in zip(self.critic.parameters(), self.critic_targ.parameters()):
                p_targ.data.mul_(1 - self.tau)
                p_targ.data.add_(self.tau * p.data)
            for p, p_targ in zip(self.actor.parameters(), self.actor_targ.parameters()):
                p_targ.data.mul_(1 - self.tau)
                p_targ.data.add_(self.tau * p.data)

class HERReplayBuffer:
    """
    Stores full episodes, relabels with HER on sampling.
    """
    def __init__(self, obs_dim, goal_dim, act_dim, size, her_ratio=0.8, future_k=4):
        self.size = 0
        self.max_size = size
        self.her_ratio = her_ratio
        self.future_k = future_k

        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.act_dim = act_dim

        self.storage = []
        self.ptr = 0

    def start_episode(self):
        self.ep_obs = []
        self.ep_ag = []
        self.ep_g = []
        self.ep_act = []
        self.ep_rew = []
        self.ep_done = []

    def store_step(self, obs, ag, g, act, rew, done, obs_next, ag_next):
        self.ep_obs.append(obs)
        self.ep_ag.append(ag)
        self.ep_g.append(g)
        self.ep_act.append(act)
        self.ep_rew.append(rew)
        self.ep_done.append(done)
        self.last_obs = obs_next
        self.last_ag = ag_next

    def end_episode(self):
        # push the episode transitions to storage
        T = len(self.ep_obs)
        episode = dict(
            o=np.array(self.ep_obs, dtype=np.float32),
            ag=np.array(self.ep_ag, dtype=np.float32),
            g=np.array(self.ep_g, dtype=np.float32),
            a=np.array(self.ep_act, dtype=np.float32),
            r=np.array(self.ep_rew, dtype=np.float32),
            d=np.array(self.ep_done, dtype=np.float32),
            o2=np.vstack([self.ep_obs[1:], self.last_obs]),
            ag2=np.vstack([self.ep_ag[1:], self.last_ag])
        )
        self.storage.append(episode)
        if len(self.storage) > self.max_size:
            self.storage.pop(0)
        self.size = min(self.size + T, self.max_size)

    def sample_batch(self, batch_size):
        # sample episodes, then sample time-steps inside each
        o, g, a, r, o2, g2, d = [], [], [], [], [], [], []

        for _ in range(batch_size):
            ep = self.storage[np.random.randint(len(self.storage))]
            T = len(ep["o"])
            idx = np.random.randint(T)
            use_her = np.random.rand() < self.her_ratio

            obs = ep["o"][idx]
            ag  = ep["ag"][idx]
            goal = ep["g"][idx]
            act = ep["a"][idx]
            rew = ep["r"][idx]
            done = ep["d"][idx]
            obs2 = ep["o2"][idx]
            ag2  = ep["ag2"][idx]

            final_goal = goal
            if use_her:
                # pick a future achieved goal as new goal
                future_idx = np.random.randint(idx, T)
                final_goal = ep["ag"][future_idx]
                # recompute reward with new goal (sparse style)
                dist = np.linalg.norm(ag2 - final_goal)
                rew = -1.0
                done = 0.0
                if dist < 0.1:
                    rew = 0.0
                    done = 1.0

            o.append(obs)
            g.append(final_goal)
            a.append(act)
            r.append(rew)
            o2.append(obs2)
            g2.append(final_goal)  # same goal next
            d.append(done)

        return (
            np.array(o), np.array(g), np.array(a),
            np.array(r)[:, None], np.array(o2),
            np.array(g2), np.array(d)[:, None]
        )
