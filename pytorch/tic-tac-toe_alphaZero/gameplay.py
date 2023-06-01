import random
from typing import Optional

import numpy as np
import numpy.typing as npt

board_type = npt.NDArray[np.int8]


def availableMoves(board: board_type) -> list[tuple[int, int]]:
    returnMoves = []
    for i, row in enumerate(board):
        for j, col in enumerate(row):
            if col == 0:
                returnMoves.append((i, j))
    return returnMoves


def makeMove(board: board_type, player: int,
             move: Optional[tuple[int, int]]) -> None:
    if not move:
        return
    board[move[0]][move[1]] = player


def randomMove(moves: list[tuple[int, int]]) -> tuple[int, int]:
    return random.choice(moves)


def gameEnd(board: board_type) -> int:
    COLUMN_COUNT = 3
    ROW_COUNT = 3

    # Check horizontal locations for win
    for r in range(ROW_COUNT):
        first_spot = board[r][0]
        if first_spot == 0:
            continue
        if first_spot == board[r][1] == board[r][2]:
            return winner(first_spot)

    # Check vertical locations for win
    for c in range(COLUMN_COUNT):
        first_spot = board[0][c]
        if first_spot == 0:
            continue
        if first_spot == board[1][c] == board[2][c]:
            return winner(first_spot)

    middle = board[1][1]
    if middle == 0:
        return 0

    # Check positively sloped diagonals
    if board[0][0] == middle == board[2][2]:
        return winner(middle)

    # Check negatively sloped diagonals
    if board[0][2] == middle == board[2][0]:
        return winner(middle)

    return 0


def winner(player: int) -> int:
    if player:
        return player
    return 0


def nextPlayer(player: int) -> int:
    return -1 if player == 1 else 1
