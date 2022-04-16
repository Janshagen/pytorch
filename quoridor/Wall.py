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
            pygame.draw.line(display, self.color, (self.c*w/2,
                             (self.r-1.94)*w/2), (self.c*w/2, (self.r+1.94)*w/2), width=5)
        elif self.orientation == 'horizontal':
            pygame.draw.line(display, self.color, ((self.c-1.94)*w/2,
                             self.r*w/2), ((self.c+1.94)*w/2, self.r*w/2), width=5)

    def move(self, MATRIX_SIZE, w):
        """Updates the wall according to the mouse's position"""
        pos = pygame.mouse.get_pos()
        self.c = max(min(round(pos[0]*2 / w), MATRIX_SIZE), 0)
        self.c = self.c if not self.c % 2 else self.c + 1
        self.r = max(min(round(pos[1]*2 / w), MATRIX_SIZE), 0)
        self.r = self.r if not self.r % 2 else self.r + 1

    def flip(self):
        """Flips the wall"""
        if self.orientation == 'vertical':
            self.orientation = 'horizontal'
        else:
            self.orientation = 'vertical'
