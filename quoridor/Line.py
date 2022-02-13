import pygame


class Line:
    def __init__(self, r, c, orientation):
        self.r = r
        self.c = c
        self.occ = False
        self.orientation = orientation
        self.color = (255, 255, 255)
        self.tag = 0

    def show(self, display, w):
        """Draws the lines, depending on what orientation they have and if there is a wall or not."""
        if self.orientation == 'vertical':
            if self.occ:
                pygame.draw.rect(display, self.color, pygame.Rect(self.c * w - 3, self.r * w, 6, w))
            else:
                pygame.draw.line(display, self.color, (self.c * w, self.r * w), (self.c * w, (self.r + 1) * w))
        elif self.orientation == 'horizontal':
            if self.occ:
                pygame.draw.rect(display, self.color, pygame.Rect(self.c * w, self.r * w - 3, w, 6))
            else:
                pygame.draw.line(display, self.color, (self.c * w, self.r * w), ((self.c + 1) * w, self.r * w))

    def paint(self, color, tag):
        """Passes the attributes of the wall to the line."""
        if not self.occ:
            self.occ = True
            self.color = color
            self.tag = tag
