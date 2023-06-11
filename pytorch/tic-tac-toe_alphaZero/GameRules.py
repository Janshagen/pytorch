import numpy as np
import numpy.typing as npt
from typing import TypeAlias


class TicTacToeGameState:
    GAME_NOT_OVER = 2
    DRAW = 0

    MOVE2INDEX = {(0, 0): 0, (0, 1): 1, (0, 2): 2,
                  (1, 0): 3, (1, 1): 4, (1, 2): 5,
                  (2, 0): 6, (2, 1): 7, (2, 2): 8}

    board_type: TypeAlias = npt.NDArray[np.int8]

    def __init__(self, board: board_type, player: int) -> None:
        # player is the player to make a move
        self.board = board
        self.player = player

    def copy(self) -> 'TicTacToeGameState':
        return TicTacToeGameState(self.board.copy(), self.player)

    def available_moves(self) -> list[tuple[int, int]]:
        returnMoves = []
        for i, row in enumerate(self.board):
            for j, col in enumerate(row):
                if col == 0:
                    returnMoves.append((i, j))
        return returnMoves

    def make_move(self, move: tuple[int, int]) -> None:
        self.board[move[0]][move[1]] = self.player
        self.next_player()

    def next_player(self) -> None:
        self.player = -self.player

    def game_status(self) -> int:
        COLUMN_COUNT = 3
        ROW_COUNT = 3

        if not (self.board == 0).any():
            return TicTacToeGameState.DRAW

        # Check horizontal locations for win
        for r in range(ROW_COUNT):
            player = self.board[r][0]
            if player == 0:
                continue
            if player == self.board[r][1] == self.board[r][2]:
                return player

        # Check vertical locations for win
        for c in range(COLUMN_COUNT):
            player = self.board[0][c]
            if player == 0:
                continue
            if player == self.board[1][c] == self.board[2][c]:
                return player

        middle = self.board[1][1]
        if middle == 0:
            return TicTacToeGameState.GAME_NOT_OVER

        # Check positively sloped diagonals
        if self.board[0][0] == middle == self.board[2][2]:
            return middle

        # Check negatively sloped diagonals
        if self.board[0][2] == middle == self.board[2][0]:
            return middle

        return TicTacToeGameState.GAME_NOT_OVER

    @staticmethod
    def game_over(status) -> bool:
        if status == TicTacToeGameState.GAME_NOT_OVER:
            return False
        return True

    @staticmethod
    def move2index(move) -> int:
        return TicTacToeGameState.MOVE2INDEX[move]


# class GameState:
#     def __init__(self, board: TicTacToeGameState.board_type, player: int) -> None:
#         self.board = board
#         self.player = player

#     def copy(self) -> 'GameState':
#         return GameState(self.board, self.player)
