import pygame
import sys

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
 

class Player:
    # moves in form ((row, col), (linerow to check, linecol to check))
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
            display, self.color, (self.c * w + w // 2, self.r * w + w // 2), w // 2 - 8
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
                    (255, 165, 0),
                    ((self.c + move[1]) * w + w // 2, (self.r + move[0]) * w + w // 2),
                    w // 2 - 8,
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

    def possibleFinish(self, vertical_lines, horizontal_lines, gridsize) -> bool:
        nextPos = Queue()
        triedPos = set()
        currentPos = (self.r, self.c)
        nextPos.EnQueue(currentPos)
        triedPos.add(currentPos)
        while not nextPos.isEmpty():
            currentPos = nextPos.DeQueue()
            for _, (key, v) in enumerate(self.MOVES.items()):
                move, lineCheck = v
                lines = vertical_lines if key in (pygame.K_LEFT, pygame.K_RIGHT) else horizontal_lines
                newPos = (currentPos[0] + move[0], currentPos[1] + move[1])
                if not lines[currentPos[0] + lineCheck[0]][currentPos[1] + lineCheck[1]].occ and newPos not in triedPos:
                    if self.winner2(newPos):
                        return True
                    nextPos.EnQueue(newPos)
                    triedPos.add(newPos)
        
        return False
    
    def winner2(self, pos) -> bool:
        """Checks if the player has won depending on where the goal is for that player."""
        return pos in self.goal
        
    def winner(self) -> bool:
        """Checks if the player has won depending on where the goal is for that player."""
        return (self.r, self.c) in self.goal