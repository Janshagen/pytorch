import pygame
import sys


class Player:
    MOVES = {
        pygame.K_UP: ((-1, 0), (0, 0)),
        pygame.K_DOWN: ((1, 0), (1, 0)),
        pygame.K_RIGHT: ((0, 1), (0, 1)),
        pygame.K_LEFT: ((0, -1), (0, 0)),
    }

    def __init__(self, color, r, c, goal) -> None:
        self.color = color
        self.r = r
        self.c = c
        self.goal = goal

    def show(self, display, w) -> None:
        pygame.draw.circle(
            display, self.color, (self.c * w + w // 2, self.r * w + w // 2), w // 2 - 5
        )

    def move(self, key, primaryLines, secondaryLines, players, w, display) -> bool:
        """Moves the player. If the position is occupied by an opponent
        the player is moved forward then left or right
        depending on the choice of the player. Returns True if move is successfull"""
        move, lineCheck = self.MOVES[key]
        opponents = [(player.r, player.c) for player in players[1:]]

        # move blocked by wall
        if not primaryLines[self.r + lineCheck[0]][self.c + lineCheck[1]].occ:
            self.r += move[0]
            self.c += move[1]

            # opponent in way
            if opponents.count((self.r, self.c)):
                # if jump blocked by player or wall: choose wich direction to jump.
                # otherwise jump in same direction as previous move
                if primaryLines[self.r + lineCheck[0]][
                    self.c + lineCheck[1]
                ].occ or opponents.count((self.r + move[0], self.c + move[1])):

                    # finds possible moves to make
                    newMove = self.chooseMove(
                        key, secondaryLines, opponents, w, display
                    )
                    # if no moves exists, move bakc and return False
                    if newMove == (0, 0):
                        self.r -= move[0]
                        self.c -= move[1]
                        return False
                    self.r += newMove[0]
                    self.c += newMove[1]
                    return True

                self.r += move[0]
                self.c += move[1]

            return True

    def chooseMove(self, oldKey, lines, opponents, w, display) -> tuple:
        movesToCheck = (
            (pygame.K_RIGHT, pygame.K_LEFT)
            if oldKey in (pygame.K_UP, pygame.K_DOWN)
            else (pygame.K_UP, pygame.K_DOWN)
        )

        availableMoves = []
        for key in movesToCheck:
            move, lineCheck = self.MOVES[key]
            if not lines[self.r + lineCheck[0]][
                self.c + lineCheck[1]
            ].occ and not opponents.count((self.r + move[0], self.c + move[1])):
                availableMoves.append(key)
                pygame.draw.circle(
                    display,
                    (255, 140, 0),
                    ((self.c + move[1]) * w + w // 2, (self.r + move[0]) * w + w // 2),
                    w // 2 - 5,
                )
        if not availableMoves:
            return (0, 0)

        pygame.display.flip()
        pygame.event.clear()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN and event.key in availableMoves:
                    return self.MOVES[event.key][0]

    def winner(self, gridsize) -> bool:
        """Checks if the player has won depending on where the goal is for that player."""
        direction, num = self.goal[:]
        if direction == "r" and int(num) == 0 and self.r == 0:
            return True
        elif direction == "r" and int(num) == 1 and self.r == gridsize - 1:
            return True
        elif direction == "c" and int(num) == 0 and self.c == 0:
            return True
        elif direction == "c" and int(num) == 1 and self.c == gridsize - 1:
            return True
