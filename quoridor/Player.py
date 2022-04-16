import pygame
import sys
from gameplay import makeMove


class Node:

    def __init__(self, data):
        self.data = data
        self.next = None


class Queue:

    def __init__(self):
        self.front = self.rear = None

    def isEmpty(self):
        return self.front == None

    # Method to add an item to the queue
    def EnQueue(self, item):
        temp = Node(item)

        if self.rear == None:
            self.front = self.rear = temp
            return
        self.rear.next = temp
        self.rear = temp

    # Method to remove an item from queue
    def DeQueue(self):

        if self.isEmpty():
            return
        temp = self.front
        self.front = temp.next

        if(self.front == None):
            self.rear = None

        return temp.data


def even(x: int):
    return not (x % 2)


class Player:
    # moves in form ((row, col), (linerow to check, linecol to check))
    MOVES = {
        pygame.K_UP: (-1, 0),
        pygame.K_DOWN: (1, 0),
        pygame.K_RIGHT: (0, 1),
        pygame.K_LEFT: (0, -1)
    }

    def __init__(self, num, color, r, c, goal, species='human') -> None:
        self.color = color
        self.r = r
        self.c = c
        self.goal = goal
        self.walls = []
        self.num = num
        self.species = species

    def show(self, display, w) -> None:
        w = w*9/18
        pygame.draw.circle(
            display, self.color, (self.c * w, self.r * w), 0.6*w)

        for start, end in self.walls:
            pygame.draw.line(display, self.color,
                             (start[1]*w, start[0]*w), (end[1]*w, end[0]*w), width=5)

    def move(self, key, board, players, w, display) -> bool:
        """Moves the player. If the position is occupied by an opponent
        the player is moved forward then left or right
        depending on the choice of the player. Returns True if move is successfull"""
        move = self.MOVES[key]
        opponents = [(player.r, player.c) for player in players[1:]]
        originalPos = (self.r, self.c)

        # move blocked by wall
        if not board[self.r + move[0]][self.c + move[1]]:
            self.r += move[0]*2
            self.c += move[1]*2
            if not opponents.count((self.r, self.c)):
                makeMove(
                    board, ('walk', (originalPos, (self.r, self.c))), self.num)
                return True

            # opponent in way
            # jump not blocked by player or wall
            if not board[self.r + move[0]][self.c + move[1]] and not opponents.count((self.r + move[0]*2, self.c + move[1]*2)):
                self.r += move[0]*2
                self.c += move[1]*2
                makeMove(
                    board, ('walk', (originalPos, (self.r, self.c))), self.num)
                return True

            # jump blocked by player or wall: choose direction
            newMove = self.chooseMove(key, board, opponents, w, display)
            # if no moves exists, move back and return False
            if newMove == (0, 0):
                self.r -= move[0]*2
                self.c -= move[1]*2
                return False

            self.r += newMove[0]*2
            self.c += newMove[1]*2
            makeMove(board, ('walk', (originalPos, (self.r, self.c))), self.num)
            return True

    def chooseMove(self, oldKey, board, opponents, w, display) -> tuple:
        movesToCheck = (
            (pygame.K_RIGHT, pygame.K_LEFT)
            if oldKey in (pygame.K_UP, pygame.K_DOWN)
            else (pygame.K_UP, pygame.K_DOWN)
        )

        availableMoves = []
        for key in movesToCheck:
            move = self.MOVES[key]
            if not board[self.r + move[0]][
                self.c + move[1]
            ] and not opponents.count((self.r + move[0]*2, self.c + move[1]*2)):
                availableMoves.append(key)
                pygame.draw.circle(
                    display,
                    (255, 165, 0),
                    ((self.c + move[1]*2) * w / 2,
                     (self.r + move[0]*2) * w / 2),
                    0.3*w,
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
                    return self.MOVES[event.key]

    def possibleFinish(self, board, currentPos) -> bool:
        nextPos = Queue()
        triedPos = set()
        nextPos.EnQueue(currentPos)
        triedPos.add(currentPos)
        while not nextPos.isEmpty():
            currentPos = nextPos.DeQueue()
            for move in (-1, 0), (1, 0), (0, 1), (0, -1):
                newPos = (currentPos[0] + move[0]*2, currentPos[1] + move[1]*2)
                if not board[currentPos[0] + move[0]][currentPos[1] + move[1]] and newPos not in triedPos:
                    if self.winner2(newPos):
                        return True
                    nextPos.EnQueue(newPos)
                    triedPos.add(newPos)

        return False

    def distanceToGoal(self, board, startPos) -> int:
        nextPos = Queue()
        triedPos = set()
        nextPos.EnQueue((startPos, 0))
        triedPos.add(startPos)
        while True:
            currentPos, steps = nextPos.DeQueue()
            for move in (-1, 0), (1, 0), (0, 1), (0, -1):
                obstacle = board[currentPos[0] +
                                 move[0]][currentPos[1] + move[1]]
                if obstacle != 5 and obstacle != 0:
                    steps -= 1
                newPos = (currentPos[0] + move[0]*2, currentPos[1] + move[1]*2)
                if not obstacle and newPos not in triedPos:
                    if self.winner2(newPos):
                        return steps + 1
                    nextPos.EnQueue((newPos, steps+1))
                    triedPos.add(newPos)

    def winner2(self, pos) -> bool:
        """Checks if the player has won depending on where the goal is for that player."""
        return pos in self.goal

    def winner(self) -> bool:
        """Checks if the player has won depending on where the goal is for that player."""
        return (self.r, self.c) in self.goal
