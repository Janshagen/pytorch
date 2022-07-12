import numpy as np
import random


def availableMoves(currentState: np.ndarray) -> list:
    returnMoves = []
    availableMoves_ = np.arange(7)

    for col in availableMoves_:
        if currentState[0][col] == 0:
            returnMoves.append(int(col))
    return returnMoves


def makeMove(currentState: np.ndarray, player: int, col: int) -> None:
    if type(col) != int:
        return

    for row in range(5, -1, -1):
        if currentState[row][col] == 0:
            currentState[row][col] = player
            return


def randomMove(moves: list) -> int:
    return random.choice(moves)


def gameEnd(board: np.ndarray) -> np.ndarray:
    COLUMN_COUNT = 7
    ROW_COUNT = 6
    # Check horizontal locations for win
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT):
            firstSpot = board[r][c]
            if firstSpot == 0:
                continue
            if firstSpot == board[r][c+1] == board[r][c+2] == board[r][c+3]:
                return winner(firstSpot)

    # Check vertical locations for win
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT-3):
            firstSpot = board[r][c]
            if firstSpot == 0:
                continue
            if firstSpot == board[r+1][c] == board[r+2][c] == board[r+3][c]:
                return winner(firstSpot)

    # Check positively sloped diaganols
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT-3):
            firstSpot = board[r][c]
            if firstSpot == 0:
                continue
            if firstSpot == board[r+1][c+1] == board[r+2][c+2] == board[r+3][c+3]:
                return winner(firstSpot)

    # Check negatively sloped diaganols
    for c in range(COLUMN_COUNT-3):
        for r in range(3, ROW_COUNT):
            firstSpot = board[r][c]
            if firstSpot == 0:
                continue
            if firstSpot == board[r-1][c+1] == board[r-2][c+2] == board[r-3][c+3]:
                return winner(firstSpot)

    return np.array([0, 0])


def winner(player: int) -> np.ndarray:
    if player == 1:
        return np.array([1, -1])
    elif player == -1:
        return np.array([-1, 1])


def nextPlayer(player: int) -> int:
    return -1*player
