import numpy as np
import numpy.typing as npt
from typing import TypeAlias, Optional
import torch


class Connect4GameState:
    GAME_NOT_OVER = 2
    DRAW = 0

    board_type: TypeAlias = npt.NDArray[np.int8]

    def __init__(self, board: board_type, player: int,
                 status: int = GAME_NOT_OVER) -> None:
        # player is the player to make a move
        self.board = board
        self.player = player

        self.status = status

    def copy(self) -> 'Connect4GameState':
        return Connect4GameState(self.board.copy(), self.player, self.status)

    @staticmethod
    def new_game(starting_player: Optional[int] = None) -> 'Connect4GameState':
        board = np.zeros((6, 7), dtype=np.int8)
        starting_player = starting_player if starting_player else -1
        return Connect4GameState(board, starting_player)

    def available_moves(self) -> list[int]:
        available_moves = []
        if self.game_over():
            return available_moves

        for col in range(7):
            if self.board[0][col] == 0:
                available_moves.append(col)
        return available_moves

    def make_move(self, move: int) -> None:
        row = 0
        for row in range(5, -1, -1):
            if self.board[row][move] == 0:
                self.board[row][move] = self.player
                break
        self.next_player()
        self.update_status(row, move)

    def next_player(self) -> None:
        self.player = -self.player

    def update_status(self, row: int, move: int) -> None:
        self.status = self.calculate_game_status(row, move)

    def calculate_game_status(self, row: int, move: int) -> int:
        start_row = max(row-3, 0)
        end_row = min(row+1, 3)
        start_column = max(move-3, 0)
        end_column = min(move+1, 4)

        # Check horizontal locations for win
        for c in range(start_column, end_column):
            player = self.board[row][c]
            if player == 0:
                continue
            if player == self.board[row][c+1] == \
                    self.board[row][c+2] == self.board[row][c+3]:
                return player

        # Check vertical locations for win
        for r in range(start_row, end_row):
            player = self.board[r][move]
            if player == 0:
                continue
            if player == self.board[r+1][move] == \
                    self.board[r+2][move] == self.board[r+3][move]:
                return player

        # Check negatively sloped diagonals
        for c in range(start_column, end_column):
            for r in range(start_row, end_row):
                player = self.board[r][c]
                if player == 0:
                    continue
                if player == self.board[r+1][c+1] == \
                        self.board[r+2][c+2] == self.board[r+3][c+3]:
                    return player

        # Check positively sloped diagonals
        for c in range(start_column, end_column):
            for r in range(start_row+3, end_row+3):
                player = self.board[r][c]
                if player == 0:
                    continue
                if player == self.board[r-1][c+1] == \
                        self.board[r-2][c+2] == self.board[r-3][c+3]:
                    return player

        if not (self.board == 0).any():
            return Connect4GameState.DRAW

        return Connect4GameState.GAME_NOT_OVER

    def game_over(self) -> bool:
        if self.status == Connect4GameState.GAME_NOT_OVER:
            return False
        return True

    def get_status(self):
        return self.status

    @staticmethod
    def get_masks(boards: torch.Tensor) -> torch.Tensor:
        number_of_states = boards.shape[0]
        ones = torch.ones((1, 7), device=boards.device)
        masks = torch.ones((number_of_states, 7), device=boards.device)

        # ändra så att rätt lager maskeras
        player_one = boards[:, 0, 0] == ones
        player_two = boards[:, 1, 0] == ones

        illegal_indices = (player_one + player_two)
        masks[illegal_indices] = 0
        return masks
