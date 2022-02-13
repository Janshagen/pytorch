import pygame


class Cell:
    def __init__(self, i, j, w):
        self.occ = False
        self.i = i
        self.j = j
        self.w = w
        self.color = (0, 0, 0)

    def show(self, screen):
        pygame.draw.rect(screen, self.color, (self.j * self.w, self.i * self.w, self.w, self.w))
