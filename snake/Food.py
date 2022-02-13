import pygame
import random


class Food:
    def __init__(self, pos, w, color, gridsize):
        self.pos = pos
        self.w = w
        self.color = color
        self.gridsize = gridsize

    def show(self, screen):
        """Draws the food"""
        pygame.draw.rect(screen, self.color, (self.pos[0] * self.w + 1, self.pos[1] * self.w + 1,
                                              self.w - 1, self.w - 1))

    def add_food(self, snake):
        """Add new food at a random location until it isn't under the snake"""
        self.pos[0] = random.randint(0, self.gridsize - 1)
        self.pos[1] = random.randint(0, self.gridsize - 1)
        while snake.body[0:snake.len].count(self.pos):
            self.pos[0] = random.randint(0, self.gridsize - 1)
            self.pos[1] = random.randint(0, self.gridsize - 1)
