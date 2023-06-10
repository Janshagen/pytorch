import random
from typing import Optional

import numpy as np
import numpy.typing as npt

board_type = npt.NDArray[np.int8]

GAME_NOT_OVER = 2
DRAW = 0


MOVE2INDEX = {(0, 0): 0, (0, 1): 1, (0, 2): 2,
              (1, 0): 3, (1, 1): 4, (1, 2): 5,
              (2, 0): 6, (2, 1): 7, (2, 2): 8}


def move2index(move) -> int:
    return MOVE2INDEX[move]


def available_moves(board: board_type) -> list[tuple[int, int]]:
    returnMoves = []
    for i, row in enumerate(board):
        for j, col in enumerate(row):
            if col == 0:
                returnMoves.append((i, j))
    return returnMoves


def make_move(board: board_type, player: int,
              move: Optional[tuple[int, int]]) -> None:
    if not move:
        return
    board[move[0]][move[1]] = player


def random_move(moves: list[tuple[int, int]]) -> tuple[int, int]:
    return random.choice(moves)


def game_status(board: board_type) -> int:
    COLUMN_COUNT = 3
    ROW_COUNT = 3

    if not (board == 0).any():
        return DRAW

    # Check horizontal locations for win
    for r in range(ROW_COUNT):
        first_spot = board[r][0]
        if first_spot == 0:
            continue
        if first_spot == board[r][1] == board[r][2]:
            return first_spot

    # Check vertical locations for win
    for c in range(COLUMN_COUNT):
        first_spot = board[0][c]
        if first_spot == 0:
            continue
        if first_spot == board[1][c] == board[2][c]:
            return first_spot

    middle = board[1][1]
    if middle == 0:
        return GAME_NOT_OVER

    # Check positively sloped diagonals
    if board[0][0] == middle == board[2][2]:
        return middle

    # Check negatively sloped diagonals
    if board[0][2] == middle == board[2][0]:
        return middle

    return GAME_NOT_OVER


def game_over(status) -> bool:
    if status == GAME_NOT_OVER:
        return False
    return True


def next_player(player: int) -> int:
    return -1 * player
