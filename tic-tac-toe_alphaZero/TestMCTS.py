import unittest
from GameRules import TicTacToeGameState
import numpy as np
import numpy.testing as npt


class TestGameState(unittest.TestCase):

    def setUp(self):
        self.blanc_game_state = TicTacToeGameState.new_game(starting_player=1)

        self.populated_boards = np.array([[
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0]], [

            [1, 0, -1],
            [0, 0, 0],
            [0, 0, 0]], [

            [1, 0, -1],
            [0, 0, 0],
            [0, 0, 1]], [

            [1, 0, -1],
            [0, -1, 0],
            [0, 0, 1]]],
            dtype=np.int8)

        self.diagonal_win = np.array([
            [1, -1, -1],
            [0, 1, 0],
            [0, 0, 1]],
            dtype=np.int8)
        self.diagonal_win_state = TicTacToeGameState(self.diagonal_win, -1)

        self.horizontal_win = np.array([
            [1, 1, 1],
            [0, -1, 0],
            [0, -1, -1]],
            dtype=np.int8)
        self.horizontal_win_state = TicTacToeGameState(self.horizontal_win, -1)

        self.vertical_win = np.array([
            [-1, 1, -1],
            [0, 1, 0],
            [0, 1, -1]],
            dtype=np.int8)
        self.vertical_win_state = TicTacToeGameState(self.vertical_win, -1)

        self.draw = np.array([
            [1, -1, 1],
            [-1, 1, 1],
            [-1, 1, -1]],
            dtype=np.int8)
        self.draw_state = TicTacToeGameState(self.draw, -1)

    def test_find_move(self):
        pass

    def test_traverse_tree(self):
        pass

    def test_expand_tree(self):
        pass

    def test_maximum_time_exceeded(self):
        pass

    def test_copy(self) -> None:
        self.assertEqual(self.blanc_game_state.player,
                         self.blanc_game_state.copy().player)
        npt.assert_array_equal(self.blanc_game_state.board,
                               self.blanc_game_state.copy().board)

    def test_available_moves(self) -> None:
        self.blanc_game_state.board = self.populated_boards[-1]
        self.assertEqual([(0, 1), (1, 0), (1, 2), (2, 0), (2, 1)],
                         self.blanc_game_state.available_moves())

    def test_make_move(self) -> None:
        self.blanc_game_state.make_move((0, 0))
        npt.assert_array_equal(self.populated_boards[0], self.blanc_game_state.board)
        self.assertEqual(-1, self.blanc_game_state.player)

        self.blanc_game_state.make_move((0, 2))
        npt.assert_array_equal(self.populated_boards[1], self.blanc_game_state.board)
        self.assertEqual(1, self.blanc_game_state.player)

        self.blanc_game_state.make_move((2, 2))
        npt.assert_array_equal(self.populated_boards[2], self.blanc_game_state.board)
        self.assertEqual(-1, self.blanc_game_state.player)

        self.blanc_game_state.make_move((1, 1))
        npt.assert_array_equal(self.populated_boards[3], self.blanc_game_state.board)
        self.assertEqual(1, self.blanc_game_state.player)

    def test_next_player(self) -> None:
        self.assertEqual(1, self.blanc_game_state.player)
        self.blanc_game_state.next_player()
        self.assertEqual(-1, self.blanc_game_state.player)
        self.blanc_game_state.next_player()
        self.assertEqual(1, self.blanc_game_state.player)

    def test_game_status(self) -> None:
        self.assertEqual(1, self.diagonal_win_state.calculate_game_status())
        self.diagonal_win_state.board = -self.diagonal_win_state.board
        self.assertEqual(-1, self.diagonal_win_state.calculate_game_status())

        self.assertEqual(1, self.vertical_win_state.calculate_game_status())
        self.vertical_win_state.board = -self.vertical_win_state.board
        self.assertEqual(-1, self.vertical_win_state.calculate_game_status())

        self.assertEqual(1, self.horizontal_win_state.calculate_game_status())
        self.horizontal_win_state.board = -self.horizontal_win_state.board
        self.assertEqual(-1, self.horizontal_win_state.calculate_game_status())

        self.assertEqual(0, self.draw_state.calculate_game_status())


if __name__ == '__main__':
    unittest.main()
