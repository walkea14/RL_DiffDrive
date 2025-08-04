import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow
import numpy as np
import imageio
from io import BytesIO

class LivePlot2D:
    def __init__(self, world_size=5.0, obstacles=None, trail_len=None):
        self.world_size = world_size
        self.obstacles = obstacles if obstacles else []
        self.trail = []
        self.trail_len = trail_len

        self.agent_pos = (0, 0)
        self.agent_heading = 0
        self.goal_pos = (1, 1)


        self._frames = []

        self.fig, self.ax = plt.subplots()
        self.agent_dot = None
        self.agent_arrow = None
        self.goal_circle = None
        self.obstacle_patches = []
        self.trail_line, = self.ax.plot([], [], 'b-', linewidth=1, alpha=0.7)

        self._setup()

    def _setup(self):
        self.ax.clear()
        self.ax.set_xlim(-self.world_size, self.world_size)
        self.ax.set_ylim(-self.world_size, self.world_size)
        self.ax.set_aspect('equal')
        self.ax.set_title("2D Navigation")

        for (ox, oy, r) in self.obstacles:
            circle = Circle((ox, oy), r, color='red', alpha=0.3)
            self.ax.add_patch(circle)
            self.obstacle_patches.append(circle)

        self.agent_dot, = self.ax.plot([], [], 'bo', markersize=8)
        self.goal_circle = Circle(self.goal_pos, 0.1, color='green', fill=True)
        self.ax.add_patch(self.goal_circle)

    def update(self, state, goal):
        x, y, theta = state
        gx, gy = goal

        self.agent_pos = (x, y)
        self.agent_heading = theta
        self.goal_pos = (gx, gy)

        self.agent_dot.set_data([x], [y])

        if self.agent_arrow:
            self.agent_arrow.remove()
        dx = 0.2 * np.cos(theta)
        dy = 0.2 * np.sin(theta)
        self.agent_arrow = Arrow(x, y, dx, dy, width=0.05, color='blue')
        self.ax.add_patch(self.agent_arrow)

        self.goal_circle.center = (gx, gy)

        self.trail.append((x, y))
        if self.trail_len is not None and len(self.trail) > self.trail_len:
            self.trail = self.trail[-self.trail_len:]

        if len(self.trail) >= 2:
            tx, ty = zip(*self.trail)
            self.trail_line.set_data(tx, ty)

    def render(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        #buf = BytesIO()
        #self.fig.savefig(buf, format='png', dpi=100)
        #buf.seek(0)
        #self._frames.append(imageio.v2.imread(buf))
        #buf.close()

        plt.pause(0.001)

    def reset_trail(self):
        self.trail = []
        self.trail_line.set_data([], [])

    def save_gif(self, filename="trajectory.gif"):
        if not self._frames:
            print("Warning: No frames to save.")
            return
        imageio.mimsave(filename, self._frames, duration=0.1)
        print(f"Saved GIF to {filename}")
