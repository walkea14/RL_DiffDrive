# envs/goal_nav_2d.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

class GoalNav2DEnv(gym.Env):
    """
    State:  [x, y, theta]
    Goal:   [gx, gy] (passed in via 'desired_goal' key, HER-style)
    Action: [v, w]   (linear vel, angular vel)
    Reward: sparse (success) + shaped (-distance)
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        world_size=5.0,
        dt=0.1,
        max_steps=200,
        success_radius=0.1,
        v_max=0.5,
        w_max=1.5,
        seed=None,
        continuous_goals=True
    ):
        super().__init__()
        self.world_size = world_size
        self.dt = dt
        self.max_steps = max_steps
        self.success_radius = success_radius
        self.v_max = v_max
        self.w_max = w_max
        self.continuous_goals = continuous_goals

        # obs = dict for HER-style: observation, desired_goal, achieved_goal
        obs_high = np.array([world_size, world_size, np.pi], dtype=np.float32)
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(-obs_high, obs_high, dtype=np.float32),
            "desired_goal": spaces.Box(-world_size, world_size, shape=(2,), dtype=np.float32),
            "achieved_goal": spaces.Box(-world_size, world_size, shape=(2,), dtype=np.float32),
        })
        self.action_space = spaces.Box(
            low=np.array([-v_max, -w_max], dtype=np.float32),
            high=np.array([ v_max,  w_max], dtype=np.float32),
            dtype=np.float32
        )

        self._rng = np.random.default_rng(seed)
        self.state = None
        self.goal = None
        self.steps = 0

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)

    def _sample_state(self):
        x = self._rng.uniform(-self.world_size, self.world_size)
        y = self._rng.uniform(-self.world_size, self.world_size)
        th = self._rng.uniform(-np.pi, np.pi)
        return np.array([x, y, th], dtype=np.float32)

    def _sample_goal(self):
        gx = self._rng.uniform(-self.world_size, self.world_size)
        gy = self._rng.uniform(-self.world_size, self.world_size)
        return np.array([gx, gy], dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.steps = 0
        self.state = self._sample_state()
        self.goal = self._sample_goal()
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        v, w = np.clip(action, self.action_space.low, self.action_space.high)

        # Unpack state
        x, y, th = self.state

        # Differential drive kinematics (unicycle)
        x  = x + v * np.cos(th) * self.dt
        y  = y + v * np.sin(th) * self.dt
        th = wrap_angle(th + w * self.dt)

        self.state = np.array([x, y, th], dtype=np.float32)

        # Compute reward
        dist = np.linalg.norm(self.goal - self.state[:2])
        success = dist < self.success_radius
        reward = -dist  # dense shaping
        if success:
            reward += 1.0

        terminated = success
        truncated = self.steps >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {"is_success": success}

    def _get_obs(self):
        obs = self.state.copy()
        ag = self.state[:2].copy()
        return {
            "observation": obs,
            "desired_goal": self.goal.copy(),
            "achieved_goal": ag
        }

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        r = -(d > self.success_radius).astype(np.float32)
        return r
