import torch
from GameRules import Connect4GameState
import random
from MCTS import MCTS
from Connect4Model import AlphaZero
import torch.nn as nn
from datetime import datetime

GAMES_FOLDER = '/home/anton/skola/egen/pytorch/connect4_alphaZero/games/'

NUMBER_OF_GAMES_TO_SAVE = 1_000

UCB1 = 2
SIMULATIONS = 50


class GameSimulator:
    def __init__(self, model: AlphaZero, UCB1: float, simulations: int) -> None:
        self.mcts = MCTS(model, UCB1, sim_number=simulations)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model

    def create_and_save_data(self, number_games: int, games_file: str) -> None:
        data = self.create_N_data_points(number_games)
        self.save_data(data, games_file)

    def create_N_data_points(self, number_games: int = 1) -> list[torch.Tensor]:
        boards = torch.tensor([], device=self.device)
        results = torch.tensor([], device=self.device)
        visits = torch.tensor([], device=self.device)

        for sample in range(number_games):
            game_boards, game_result, game_visits = self.get_game_data()

            boards = torch.cat((boards, game_boards), dim=0)
            results = torch.cat((results, game_result), dim=0)
            visits = torch.cat((visits, game_visits), dim=0)

            if (sample+1) % 100 == 0:
                time = datetime.today().strftime("%H:%M")
                print(f"{sample+1} games done at {time}")

        visits = self.reshape_and_normalize(visits)
        return [boards, results, visits]

    def get_game_data(self) -> tuple[torch.Tensor, ...]:
        game_state = Connect4GameState.new_game(random.choice([-1, 1]))
        all_boards = torch.tensor([], device=self.device)
        player_turns = torch.tensor([], device=self.device)

        all_visits = torch.tensor([], device=self.device)
        while True:
            game_state = self.find_and_make_move(game_state)
            player_turns = self.add_player_turns(game_state, player_turns)
            all_boards = self.add_new_flipped_boards(game_state, all_boards)
            all_visits = self.add_new_flipped_visits(all_visits)

            if game_state.game_over():
                # ändra så att varannan är 1 och varanan -1
                results = game_state.get_status() * player_turns
                results = results.unsqueeze(dim=1)
                return all_boards, results, all_visits

    def find_and_make_move(self, game_state: Connect4GameState) -> Connect4GameState:
        move = self.mcts.find_move(game_state)
        game_state.make_move(move)
        return game_state

    def add_new_flipped_boards(self,
                               game_state: Connect4GameState,
                               all_boards: torch.Tensor) -> torch.Tensor:
        torch_board = self.model.state2tensor(game_state)
        all_boards = self.add_flipped_states(all_boards, torch_board, flip_dims=(3,))
        return all_boards

    def add_new_flipped_visits(self, all_visits: torch.Tensor) -> torch.Tensor:
        visits = torch.zeros((1, 7), dtype=torch.float32, device=self.device)
        for child in self.mcts.root.children:
            visits[0][child.move] = child.visits

        all_visits = self.add_flipped_states(all_visits, visits, flip_dims=(1,))
        return all_visits

    def add_flipped_states(self,
                           stack: torch.Tensor,
                           new_state: torch.Tensor,
                           flip_dims: tuple) -> torch.Tensor:
        stack = torch.cat((stack, new_state), dim=0)
        new_state = torch.flip(new_state, dims=flip_dims)
        stack = torch.cat((stack, new_state), dim=0)
        return stack

    def add_player_turns(self, game_state: Connect4GameState,
                         player_turns: torch.Tensor) -> torch.Tensor:
        player = torch.tensor([game_state.player, game_state.player],
                              dtype=torch.float32, device=self.device)
        player_turns = torch.cat((player_turns, player))
        return player_turns

    def reshape_and_normalize(self, input: torch.Tensor) -> torch.Tensor:
        input = input.reshape((-1, 7))
        input = nn.functional.normalize(input, dim=1, p=1)
        return input

    def save_data(self, data: list[torch.Tensor], games_file: str) -> None:
        with open(games_file, 'wb') as file:
            torch.save(data, file)


if __name__ == '__main__':
    game_simulator = GameSimulator(AlphaZero(), UCB1, SIMULATIONS)
    for i in range(10):
        games_file = GAMES_FOLDER + f"game_{i}.pt"
        game_simulator.create_and_save_data(NUMBER_OF_GAMES_TO_SAVE, games_file)
