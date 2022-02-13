import pygame


class Wall:
    def __init__(self, color):
        self.r = 0
        self.c = 0
        self.orientation = 'horizontal'
        self.color = color

    def show(self, display, w):
        """Draws the wall"""
        if self.orientation == 'vertical':
            pygame.draw.rect(display, self.color, pygame.Rect(self.c * w - 3, (self.r - 1) * w, 6, 2 * w))
        elif self.orientation == 'horizontal':
            pygame.draw.rect(display, self.color, pygame.Rect((self.c - 1) * w, self.r * w - 3, 2 * w, 6))

    def move(self, gridsize, w):
        """Updates the wall according to the mouse's position"""
        pos = pygame.mouse.get_pos()
        self.c = max(min(round(pos[0] / w), gridsize - 1), 1)
        self.r = max(min(round(pos[1] / w), gridsize - 1), 1)

    def flip(self):
        """Flips the wall"""
        if self.orientation == 'vertical':
            self.orientation = 'horizontal'
        else:
            self.orientation = 'vertical'
