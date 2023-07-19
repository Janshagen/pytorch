import torch
from GameRules import TicTacToeGameState
import random
from MCTS import MCTS
from TicTacToeModel import AlphaZero
import torch.nn as nn
from datetime import datetime

GAMES_FILE = '/home/anton/skola/egen/pytorch/tic-tac-toe_alphaZero/games.pt'

NUMBER_OF_GAMES_TO_SAVE = 10_000

UCB1 = 1.4
SIMULATIONS = 100


class GameSimulator:
    def __init__(self, model: AlphaZero, UCB1: float, simulations: int) -> None:
        self.mcts = MCTS(model, UCB1, sim_number=simulations)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model

    def create_and_save_data(self, number_games: int) -> None:
        data = self.create_N_data_points(number_games)
        self.save_data(data)

    def create_N_data_points(self, number_games: int) -> list[torch.Tensor]:
        boards = torch.tensor([], device=self.device)
        results = torch.tensor([], device=self.device)
        visits = torch.tensor([], device=self.device)
        game_lengths = torch.zeros((number_games,))

        for sample in range(number_games):
            game_boards, game_result, game_visits = self.get_game_data()
            game_lengths[sample] = game_boards.shape[0]

            boards = torch.cat((boards, game_boards), dim=0)
            results = torch.cat((results, game_result), dim=0)
            visits = torch.cat((visits, game_visits), dim=0)

            if (sample+1) % 1000 == 0:
                time = datetime.today().strftime("%H:%M")
                print(f"{sample+1} games done at {time}")

        visits = self.reshape_and_normalize(visits)
        return [boards, results, visits, game_lengths]

    def get_game_data(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        game_board_states, game_result, game_visits = self.game()

        num_moves = game_board_states.shape[0]
        game_result = game_result.expand((num_moves, 1))
        return game_board_states, game_result, game_visits

    def game(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        game_state = TicTacToeGameState.new_game(random.choice([1, -1]))
        all_boards = torch.zeros((4, 4, 3, 3), device=self.device)

        all_visits = torch.tensor([], device=self.device)
        while True:
            game_state = self.find_and_make_move(game_state)
            all_boards = self.add_new_flipped_boards(game_state, all_boards)
            all_visits = self.add_new_flipped_visits(all_visits)

            if game_state.game_over():
                result = torch.tensor([game_state.get_status()], dtype=torch.float32,
                                      device=self.device)
                return all_boards, result, all_visits

    def find_and_make_move(self, game_state: TicTacToeGameState) -> TicTacToeGameState:
        move = self.mcts.find_move(game_state)
        game_state.make_move(move)
        return game_state

    def add_new_flipped_boards(self,
                               game_state: TicTacToeGameState,
                               all_boards: torch.Tensor) -> torch.Tensor:
        torch_board = self.model.state2tensor(game_state)
        all_boards = self.add_flipped_states(all_boards, torch_board, flip_dims=(2, 3))
        return all_boards

    def add_new_flipped_visits(self, all_visits: torch.Tensor) -> torch.Tensor:
        visits = torch.zeros((1, 3, 3), dtype=torch.float32, device=self.device)
        for child in self.mcts.root.children:
            visits[0][child.move] = child.visits

        all_visits = self.add_flipped_states(all_visits, visits, flip_dims=(1, 2))
        return all_visits

    def add_flipped_states(self,
                           stack: torch.Tensor,
                           new_state: torch.Tensor,
                           flip_dims: tuple) -> torch.Tensor:
        for i in range(4):
            permutation = new_state.rot90(k=i, dims=flip_dims)
            stack = torch.cat((stack, permutation), dim=0)
        return stack

    def reshape_and_normalize(self, input: torch.Tensor) -> torch.Tensor:
        input = input.reshape((-1, 9))
        input = nn.functional.normalize(input, dim=1, p=1)
        return input

    def save_data(self, data: list[torch.Tensor]) -> None:
        with open(GAMES_FILE, 'wb') as file:
            torch.save(data, file)


if __name__ == '__main__':
    game_simulator = GameSimulator(AlphaZero(), UCB1, SIMULATIONS)
    game_simulator.create_and_save_data(NUMBER_OF_GAMES_TO_SAVE)
