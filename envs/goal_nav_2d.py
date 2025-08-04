import gymnasium as gym
from gymnasium import spaces
import numpy as np

def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

class GoalNav2DEnv(gym.Env):
    """
    State:  [x, y, theta]
    Goal:   [gx, gy]
    Action: [v, w]
    Reward: +1 for reaching goal, -dist for progress, -10 for collision
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
        obstacles=None,  # NEW
        collision_penalty=-10.0,  # NEW
        terminate_on_collision=True,  # NEW
        seed=None
    ):
        super().__init__()
        self.world_size = world_size
        self.dt = dt
        self.max_steps = max_steps
        self.success_radius = success_radius
        self.v_max = v_max
        self.w_max = w_max

        self.obstacles = obstacles if obstacles else []
        self.collision_penalty = collision_penalty
        self.terminate_on_collision = terminate_on_collision

        self._rng = np.random.default_rng(seed)

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

        self.state = None
        self.goal = None
        self.steps = 0

    def _sample_free_position(self):
        while True:
            x = self._rng.uniform(-self.world_size, self.world_size)
            y = self._rng.uniform(-self.world_size, self.world_size)
            collision = any(np.hypot(x - ox, y - oy) < r for (ox, oy, r) in self.obstacles)
            if not collision:
                return np.array([x, y], dtype=np.float32)

    def _sample_state(self):
        xy = self._sample_free_position()
        th = self._rng.uniform(-np.pi, np.pi)
        return np.array([*xy, th], dtype=np.float32)

    def _sample_goal(self):
        return self._sample_free_position()


    def reset(self, seed=None, options=None):
        multigoal = True
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.steps = 0

        #Random vs set start/goal toggle
        #self.state = self._sample_state()
        #self.goal = self._sample_goal()
        self.state = np.array([-4.0, -4.0, 0.0], dtype=np.float32)
        self.goal = np.array([4.0, -4.0], dtype=np.float32)

        return self._get_obs(), {}
        ep_obs = env.reset()[0]
        if not multi_goal:
            env.goal = fixed_goal.copy()
            ep_obs["desired_goal"] = fixed_goal.copy()


    def step(self, action):
        self.steps += 1
        v, w = np.clip(action, self.action_space.low, self.action_space.high)

        x, y, th = self.state
        x_new = x + v * np.cos(th) * self.dt
        y_new = y + v * np.sin(th) * self.dt
        th_new = wrap_angle(th + w * self.dt)
        new_state = np.array([x_new, y_new, th_new], dtype=np.float32)

        # Check for collision
        collided = self._check_collision(x_new, y_new)

        # Check if reached goal
        dist_to_goal = np.linalg.norm(self.goal - new_state[:2])
        reached_goal = dist_to_goal < self.success_radius
        old_distance = np.linalg.norm(self.goal - self.state[:2])
        new_distance = np.linalg.norm(self.goal - new_state[:2])


        if collided:
            reward = self.collision_penalty
            terminated = self.terminate_on_collision
            truncated = not terminated
        elif reached_goal:
            reward = 200
            terminated = True
            truncated = False
        else:
            reward = 0
            if (old_distance-new_distance) > 0:
                reward += 1
            else:
                reward = -2.5
            #reward += 2.4 * (old_distance - new_distance)

            # Alignment
            goal_vec = self.goal - self.state[:2]
            goal_dir = goal_vec / (np.linalg.norm(goal_vec) + 1e-6)
            heading = np.array([np.cos(new_state[2]), np.sin(new_state[2])])
            alignment = np.dot(goal_dir, heading)
            reward += alignment * 0.6

            #Velocity reward
            reward += 0.2*v

            # Time penalty
            #reward -= 0.04

            #Excessive turn penalty
            reward -= 0.6 * abs(w) ** 2

            #Penalize reverse motion
            if v < 0:
                reward -= 0.5

            # Goal bonus
            if reached_goal:
                reward += 200  # distance shaping
            terminated = False
            truncated = False

        self.state = new_state
        return self._get_obs(), reward, terminated, truncated, {"is_success": reached_goal, "collision": collided}

    def _get_obs(self):
        obs = self.state.copy()
        ag = self.state[:2].copy()
        return {
            "observation": obs,
            "desired_goal": self.goal.copy(),
            "achieved_goal": ag
        }

    def _check_collision(self, x, y):
        for ox, oy, r in self.obstacles:
            if np.hypot(x - ox, y - oy) < r:
                return True
        return False

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return (d < self.success_radius).astype(np.float32) * 1.0
