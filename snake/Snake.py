import pygame
from Food import Food


class Snake(Food):
    def __init__(self, pos, w, vel, color, gridsize):
        super().__init__(pos, w, color, gridsize)
        self.body = [pos]
        self.len = 2
        self.vel = vel
        self.w = w
        self.color = color
        self.gridsize = gridsize

    def show(self, screen):
        """Draws the snake, all the snakes recent locations are stored, but only drawn if the snake is that long"""
        for pos in self.body[0:self.len]:
            pygame.draw.rect(screen, self.color, (pos[0] * self.w + 1, pos[1] * self.w + 1, self.w - 1, self.w - 1))

    def update(self):
        """Moves the snake one square in the direction of its velocity. All the snakes recent locations are stored until
        that list is longer than the gridsize squared."""
        pos = [0, 0]
        pos[0] = self.body[0][0] + self.vel[0]
        pos[1] = self.body[0][1] + self.vel[1]
        self.body.insert(0, pos)
        if len(self.body) > self.gridsize**2:
            self.body.pop(-1)

    def turn(self):
        """Changes the snakes velocity"""
        for event in pygame.event.get(pygame.KEYDOWN):
            if event.key == pygame.K_LEFT and not self.vel == [1, 0]:
                self.vel = [-1, 0]
                return
            elif event.key == pygame.K_RIGHT and not self.vel == [-1, 0]:
                self.vel = [1, 0]
                return
            elif event.key == pygame.K_UP and not self.vel == [0, 1]:
                self.vel = [0, -1]
                return
            elif event.key == pygame.K_DOWN and not self.vel == [0, -1]:
                self.vel = [0, 1]
                return

    def eat(self, food):
        """Checks if the head of the snake is in the same position as the food"""
        if self.body[0] == food.pos:
            self.len += 1
            food.add_food(self)

    def crash(self):
        """Checks if the snake has crashed into itself or a wall"""
        if self.body[0][0] >= self.gridsize or self.body[0][0] < 0 or \
                self.body[0][1] >= self.gridsize or self.body[0][1] < 0:
            return True
        for pos in self.body[1:self.len]:
            if pos == self.body[0]:
                return True
