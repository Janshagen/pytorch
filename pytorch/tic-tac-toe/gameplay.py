import numpy as np
import random


def availableMoves(currentState: np.ndarray) -> list:
    returnMoves = []

    for i, row in enumerate(currentState):
        for j, col in enumerate(row):
            if col == 0:
                returnMoves.append((i, j))
    return returnMoves


def makeMove(currentState: np.ndarray, player: int, move: tuple) -> None:
    if not move:
        return

    currentState[move[0]][move[1]] = player


def randomMove(moves: list) -> int:
    return random.choice(moves)


def gameEnd(board: np.ndarray) -> np.ndarray:
    COLUMN_COUNT = 3
    ROW_COUNT = 3

    # Draw
    if not availableMoves(board):
        return np.array([1, 1])

    # Check horizontal locations for win
    for r in range(ROW_COUNT):
        firstSpot = board[r][0]
        if firstSpot == 0:
            continue
        if firstSpot == board[r][1] == board[r][2]:
            return winner(firstSpot)

    # Check vertical locations for win
    for c in range(COLUMN_COUNT):
        firstSpot = board[0][c]
        if firstSpot == 0:
            continue
        if firstSpot == board[1][c] == board[2][c]:
            return winner(firstSpot)

    middle = board[1][1]
    if middle == 0:
        return np.array([0, 0])

    # Check positively sloped diaganols
    if board[0][0] == middle == board[2][2]:
        return winner(middle)

    # Check negatively sloped diaganols
    if board[0][2] == middle == board[2][0]:
        return winner(middle)

    return np.array([0, 0])


def winner(player: int) -> np.ndarray:
    if player == 1:
        return np.array([1, -1])
    elif player == -1:
        return np.array([-1, 1])


def nextPlayer(player: int) -> int:
    return -1 if player == 1 else 1
