import random
import numpy as np


def availableMoves(board, players, player, MATRIX_SIZE):
    walks, placeWalls = [], []
    walks = findWalkMoves(board, player)
    placeWalls = findWallMoves(board, players, MATRIX_SIZE)

    placeWalls.extend(walks)
    return placeWalls


def findWalkMoves(board, player) -> list:
    walks = []
    currentPos = locatePlayer(board, player)

    for move in ((0, 1), (1, 0), (-1, 0), (0, -1)):
        newPos = (currentPos[0] + move[0]*2, currentPos[1] + move[1]*2)
        if board[currentPos[0] + move[0]][currentPos[1] + move[1]]:
            continue

        if not board[newPos[0]][newPos[1]]:
            walks.append(('walk', (currentPos, newPos)))
            continue

        if not board[newPos[0] + move[0]][newPos[1] + move[1]] and not board[newPos[0] + move[0]*2][newPos[1] + move[1]*2]:
            walks.append(('walk',
                          (currentPos, (newPos[0] + move[0]*2, newPos[1] + move[1]*2))))
            continue

        # kolla sidorna
        movesToCheck = ((-1, 0), (1, 0)) if move in ((0, 1),
                                                     (0, -1)) else ((0, 1), (0, -1))
        for newMove in movesToCheck:
            if not board[newPos[0] + newMove[0]][newPos[1] + newMove[1]] and not board[newPos[0] + newMove[0]*2][newPos[1] + newMove[1]*2]:
                walks.append(('walk',
                             (currentPos, (newPos[0] + newMove[0]*2, newPos[1] + newMove[1]*2))))

    return walks


def locatePlayer(board, player):
    for i, col in enumerate(board):
        for j, spot in enumerate(col):
            if spot == player:
                return (i, j)


def findWallMoves(board, players, MATRIX_SIZE):
    walls = []
    for row in range(2, MATRIX_SIZE-2, 2):
        for col in range(2, MATRIX_SIZE-2, 2):
            if board[row][col]:
                continue
            for dir in ((1, 0), (0, 1)):
                if not board[row - dir[0]][col - dir[1]] and not board[row + dir[0]][col + dir[1]] and not playerBlocked(board, players, row, col, dir):
                    walls.append(
                        ('wall', ((row, col), (row-dir[0], col-dir[1]), (row+dir[0], col+dir[1]))))
    return walls


def playerBlocked(board, players, row, col, dir) -> bool:
    blocked = False
    board[row][col] = 5
    board[row - dir[0]][col - dir[1]] = 5
    board[row + dir[0]][col + dir[1]] = 5
    for player in players:
        pos = locatePlayer(board, player.num)
        if not player.possibleFinish(board, pos):
            blocked = True
            break
    board[row][col] = 0
    board[row - dir[0]][col - dir[1]] = 0
    board[row + dir[0]][col + dir[1]] = 0
    return blocked


def placeWall(players, wall, board) -> bool:
    dir = (0, 1) if wall.orientation == "horizontal" else (1, 0)
    row = wall.r
    col = wall.c

    if not board[row][col] and not board[row - dir[0]][col - dir[1]] and not board[row + dir[0]][col + dir[1]]:
        board[row][col] = 5
        board[row - dir[0]][col - dir[1]] = 5
        board[row + dir[0]][col + dir[1]] = 5

        for player in players:
            pos = locatePlayer(board, player.num)
            if not player.possibleFinish(board, pos):
                board[row][col] = 0
                board[row - dir[0]][col - dir[1]] = 0
                board[row + dir[0]][col + dir[1]] = 0
                return False

        players[0].walls.append(
            ((row - 1.94*dir[0], col - 1.94*dir[1]), (row + 1.94*dir[0], col + 1.94*dir[1])))
        return True


def randomMove(moves):
    return random.choice(moves)


def makeMove(board, move, player):
    if not move:
        return

    if move[0] == 'wall':
        for row, col in move[1]:
            board[row][col] = 5
    else:  # move[0] == 'walk'
        oldPos = move[1][0]
        newPos = move[1][1]
        board[oldPos[0]][oldPos[1]] = 0
        board[newPos[0]][newPos[1]] = player


def gameEnd(players, currentPlayer, move):
    if move[0] == 'wall':
        return False
    for player in players:
        if player.num == currentPlayer and player.winner2(move[1][1]):
            return True
    return False


def nextPlayer(players, wall) -> None:
    """Changes order of the players list"""
    currentPlayer = players.pop(0)
    players.append(currentPlayer)
    wall.color = players[0].color


def winner(currentPlayer):
    if currentPlayer == 1:
        return np.array([1, -1])
    return np.array([-1, 1])
