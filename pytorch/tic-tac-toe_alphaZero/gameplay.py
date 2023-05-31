import numpy as np
import random
from typing import Optional
import numpy.typing as npt
board_type = npt.NDArray[np.int8]


def availableMoves(currentState: board_type) -> list[tuple[int, int]]:
    returnMoves = []
    for i, row in enumerate(currentState):
        for j, col in enumerate(row):
            if col == 0:
                returnMoves.append((i, j))
    return returnMoves


def makeMove(currentState: board_type, player: int,
             move: Optional[tuple[int, int]]) -> None:
    if not move:
        return

    currentState[move[0]][move[1]] = player


def randomMove(moves: list[tuple[int, int]]) -> tuple[int, int]:
    return random.choice(moves)


def gameEnd(board: board_type) -> np.ndarray:
    COLUMN_COUNT = 3
    ROW_COUNT = 3

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

    # Check positively sloped diagonals
    if board[0][0] == middle == board[2][2]:
        return winner(middle)

    # Check negatively sloped diagonals
    if board[0][2] == middle == board[2][0]:
        return winner(middle)

    return np.array([0, 0])


def winner(player: int) -> np.ndarray:
    if player == 1:
        return np.array([1, -1])
    elif player == -1:
        return np.array([-1, 1])
    return np.array([0, 0])


def nextPlayer(player: int) -> int:
    return -1 if player == 1 else 1
