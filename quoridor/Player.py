import pygame


def choose(player, updwn, r, c, w, display):
    if updwn:
        right = 1
        down = 0
    else:
        right = 0
        down = 1
    pygame.draw.circle(display, (255, 140, 0), ((c + right) * w + w // 2, (r + down) * w + w // 2), w // 2 - 5)
    pygame.draw.circle(display, (255, 140, 0), ((c - right) * w + w // 2, (r - down) * w + w // 2), w // 2 - 5)
    pygame.display.flip()
    event_ = pygame.event.wait()
    if event_.type == pygame.KEYDOWN:
        if updwn and event_.key == pygame.K_a:
            player.decided = True
            return -1
        elif updwn and event_.key == pygame.K_d:
            player.decided = True
            return 1
        elif not updwn and event_.key == pygame.K_w:
            player.decided = True
            return -1
        elif not updwn and event_.key == pygame.K_s:
            player.decided = True
            return 1
        else:
            return 0
    else:
        return 0


class Player:
    def __init__(self, color, r, c, goal):
        self.color = color
        self.r = r
        self.c = c
        self.current = False
        self.goal = goal
        self.decided = False

    def show(self, display, w):
        pygame.draw.circle(display, self.color, (self.c * w + w // 2, self.r * w + w // 2), w // 2 - 5)

    def move(self, event, horizontal_lines, vertical_lines, opponents, w, display):
        """Moves the player. If the position is occupied by an opponent the player is moved forward then left or right
        depending on the choice of the player."""
        if event.key == pygame.K_UP:
            if not horizontal_lines[self.r][self.c].occ:
                self.r -= 1
                if opponents.count((self.r, self.c)):
                    if not horizontal_lines[self.r][self.c].occ and not opponents.count((self.r - 1, self.c)):
                        self.r -= 1
                    else:
                        while not self.decided:
                            pygame.draw.circle(display, (255, 255, 0), (self.c * w + w // 2, self.r * w + w // 2), w // 2 - 5)
                            self.c = self.c + choose(self, True, self.r, self.c, w, display)
                        self.decided = False
                return True
        elif event.key == pygame.K_DOWN:
            if not horizontal_lines[self.r + 1][self.c].occ:
                self.r += 1
                if opponents.count((self.r, self.c)):
                    if not horizontal_lines[self.r + 1][self.c].occ and not opponents.count((self.r + 1, self.c)):
                        self.r += 1
                    else:
                        while not self.decided:
                            self.c = self.c + choose(self, True, self.r, self.c, w, display)
                        self.decided = False
                return True
        elif event.key == pygame.K_RIGHT:
            if not vertical_lines[self.r][self.c + 1].occ:
                self.c += 1
                if opponents.count((self.r, self.c)):
                    if not vertical_lines[self.r][self.c + 1].occ and not opponents.count((self.r, self.c + 1)):
                        self.c += 1
                    else:
                        while not self.decided:
                            self.r = self.r + choose(self, False, self.r, self.c, w, display)
                        self.decided = False
                return True
        elif event.key == pygame.K_LEFT:
            if not vertical_lines[self.r][self.c].occ:
                self.c -= 1
                if opponents.count((self.r, self.c)):
                    if not vertical_lines[self.r][self.c].occ and not opponents.count((self.r, self.c - 1)):
                        self.c -= 1
                    else:
                        while not self.decided:
                            self.r = self.r + choose(self, False, self.r, self.c, w, display)
                        self.decided = False
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


