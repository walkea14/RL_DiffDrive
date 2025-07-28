import pygame
import sys

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (50, 50, 255)
GREEN = (50, 200, 50)
RED = (200, 50, 50)

class GridVisualizer:
    def __init__(self, grid_size=(5, 5), cell_size=100, obstacles=None, goal=(4, 4)):
        pygame.init()
        self.grid_rows, self.grid_cols = grid_size
        self.cell_size = cell_size
        self.screen = pygame.display.set_mode(
            (self.grid_cols * cell_size, self.grid_rows * cell_size)
        )
        pygame.display.set_caption("Grid World RL")

        self.obstacles = obstacles if obstacles else []
        self.goal = goal
        self.agent_pos = (0, 0)
        self.clock = pygame.time.Clock()

    def set_agent_pos(self, pos):
        self.agent_pos = pos

    def draw_grid(self):
        self.screen.fill(WHITE)
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                rect = pygame.Rect(c * self.cell_size, r * self.cell_size,
                                   self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, GRAY, rect, 1)

                if (r, c) in self.obstacles:
                    pygame.draw.rect(self.screen, BLACK, rect)
                elif (r, c) == self.goal:
                    pygame.draw.rect(self.screen, GREEN, rect)

        # Draw agent
        r, c = self.agent_pos
        rect = pygame.Rect(c * self.cell_size + 10, r * self.cell_size + 10,
                           self.cell_size - 20, self.cell_size - 20)
        pygame.draw.rect(self.screen, BLUE, rect)

        pygame.display.flip()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def close(self):
        pygame.quit()
