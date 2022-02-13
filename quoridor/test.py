import pygame


def free(lines, opponents, r, c):
    if not lines[r][c].occ and not opponents.count((r, c)):
        return True


class Player:
    def __init__(self, color, r, c, goal):
        self.color = color
        self.r = r
        self.c = c
        self.current = False
        self.goal = goal

    def show(self, display, w):
        pygame.draw.circle(display, self.color, (self.c * w + w // 2, self.r * w + w // 2), w // 2 - 5)

    def move(self, event, horizontal_lines, vertical_lines, opponents):
        """Moves the player. If the position is occupied by an opponent the player is moved forward then left or right
        depending on if that spot is available and accessible."""
        if event.key == pygame.K_UP:
            if not horizontal_lines[self.r][self.c].occ:
                self.r -= 1
                if opponents.count((self.r, self.c)):
                    if free(horizontal_lines, opponents, self.r, self.c):
                        self.r -= 1
                    else:
                        if free(vertical_lines, opponents, self.r, self.c + 1):
                            self.c += 1
                        else:
                            self.c -= 1
                return True
        elif event.key == pygame.K_DOWN:
            if not horizontal_lines[self.r + 1][self.c].occ:
                self.r += 1
                if opponents.count((self.r, self.c)):
                    if free(horizontal_lines, opponents, self.r + 1, self.c):
                        self.r += 1
                    else:
                        if free(vertical_lines, opponents, self.r, self.c + 1):
                            self.c += 1
                        else:
                            self.c -= 1
                return True
        elif event.key == pygame.K_RIGHT:
            if not vertical_lines[self.r][self.c + 1].occ:
                self.c += 1
                if opponents.count((self.r, self.c)):
                    if free(vertical_lines, opponents, self.r, self.c + 1):
                        self.c += 1
                    else:
                        if free(horizontal_lines, opponents, self.r + 1, self.c):
                            self.r += 1
                        else:
                            self.r -= 1
                return True
        elif event.key == pygame.K_LEFT:
            if not vertical_lines[self.r][self.c].occ:
                self.c -= 1
                if opponents.count((self.r, self.c)):
                    if free(vertical_lines, opponents, self.r, self.c):
                        self.c -= 1
                    else:
                        if free(horizontal_lines, opponents, self.r + 1, self.c):
                            self.r += 1
                        else:
                            self.r -= 1
                return True

    def winner(self, gridsize):
        """Checks if the player has won depending on where the goal is for that player."""
        direction, num = self.goal[:]
        if direction == 'r' and int(num) == 0 and self.r == 0:
            return True
        elif direction == 'r' and int(num) == 1 and self.r == gridsize - 1:
            return True
        elif direction == 'c' and int(num) == 0 and self.c == 0:
            return True
        elif direction == 'c' and int(num) == 1 and self.c == gridsize - 1:
            return True

