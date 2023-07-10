import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from TrainingData import TrainingData
from GameRules import TicTacToeGameState
from MCTS import MCTS
from TicTacToeModel import AlphaZero, Loss

# Constants
LOAD_MODEL = False
SAVE_MODEL = True

LEARNING_RATE = 0.2
WEIGHT_DECAY = 0.01

N_BATCHES = 3_000
BATCH_SIZE = 5

SIMULATIONS = 100
UCB1 = 1.4


class Trainer:
    def __init__(self, load_file: Optional[str] = None) -> None:
        self.learning_data = self.create_learning_data(load_file)
        self.mcts = MCTS(self.learning_data.model, UCB1, sim_number=SIMULATIONS)

    def main(self) -> None:
        print("Training Started")
        self.train()
        self.validate()

        if SAVE_MODEL:
            self.learning_data.save_model()

    def create_learning_data(self, load_file: Optional[str] = None) -> TrainingData:
        model = self.create_model(load_file)
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=LEARNING_RATE,
                                    weight_decay=WEIGHT_DECAY)
        loss = Loss()

        learning_data = TrainingData(model, loss, optimizer, N_BATCHES)
        return learning_data

    def create_model(self, load_file: Optional[str] = None) -> AlphaZero:
        model = AlphaZero()
        if LOAD_MODEL:
            model = model.load_model(load_file)
        model.train()
        return model

    def train(self) -> None:
        for batch in range(N_BATCHES):
            boards = torch.tensor([], device=self.learning_data.device)
            results = torch.tensor([], device=self.learning_data.device)
            visits = torch.tensor([], device=self.learning_data.device)
            game_lengths = [0]*BATCH_SIZE
            for sample in range(BATCH_SIZE):
                game_boards, game_result, game_visits = self.get_game_data()
                game_lengths[sample] = game_boards.shape[0]

                boards = torch.cat((boards, game_boards), dim=0)
                results = torch.cat((results, game_result), dim=0)
                visits = torch.cat((visits, game_visits), dim=0)

            visits = self.reshape_and_normalize(visits)

            evaluations, policies = self.learning_data.model.forward(boards)
            policies = self.mask_illegal_moves(boards, policies)
            policies = self.reshape_and_normalize(policies)

            error = self.learning_data.loss.forward(
                evaluations, results, policies, visits, game_lengths
            )
            error.backward()

            self.learning_data.optimizer.step()
            self.learning_data.optimizer.zero_grad()
            self.learning_data.scheduler.step()

            if (batch + 1) % (N_BATCHES/10) == 0:
                self.print_info(batch, evaluations, results, error)
                if SAVE_MODEL:
                    self.learning_data.save_model()

    def get_game_data(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        game_board_states, game_result, game_visits = self.game()

        num_moves = game_board_states.shape[0]
        game_result = game_result.expand((num_moves, 1))
        return game_board_states, game_result, game_visits

    def reshape_and_normalize(self, input: torch.Tensor) -> torch.Tensor:
        input = input.reshape((-1, 9))
        input = nn.functional.normalize(input, dim=1, p=1)
        return input

    def mask_illegal_moves(self, boards: torch.Tensor,
                           policies: torch.Tensor) -> torch.Tensor:
        masks = TicTacToeGameState.get_masks(boards)
        return policies * masks

    def print_info(self, batch: int, evaluations: torch.Tensor,
                   result: torch.Tensor, error: torch.Tensor) -> None:
        print(
            f'Batch [{batch+1}/{N_BATCHES}], Loss: {error.item():.8f},',
            f'evaluation: {evaluations[-1].item():.4f}, result: {result[0][0].item()}')

    def game(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        game_state = TicTacToeGameState.new_game(random.choice([1, -1]))
        all_boards = self.new_boards(game_state)

        all_visits = torch.tensor([], device=self.learning_data.device)
        while True:
            game_state = self.find_and_make_move(game_state)
            all_boards = self.add_new_flipped_boards(game_state, all_boards)
            all_visits = self.add_new_flipped_visits(all_visits)

            if game_state.game_over():
                visits = torch.zeros((1, 3, 3), dtype=torch.float32,
                                     device=self.learning_data.device)
                all_visits = self.add_flipped_states(all_visits,
                                                     visits,
                                                     flip_dims=(1, 2)
                                                     )

                result = torch.tensor([game_state.get_status()], dtype=torch.float32,
                                      device=self.learning_data.device)
                return all_boards, result, all_visits

    def new_boards(self, game_state: TicTacToeGameState) -> torch.Tensor:
        all_boards = torch.zeros((4, 4, 3, 3), device=self.learning_data.device)
        for i in range(4):
            if game_state.player == 1:
                all_boards[i][2] = torch.ones((3, 3))
            if game_state.player == -1:
                all_boards[i][3] = torch.ones((3, 3))
        return all_boards

    def find_and_make_move(self, game_state: TicTacToeGameState) -> TicTacToeGameState:
        move = self.mcts.find_move(game_state)
        game_state.make_move(move)
        return game_state

    def add_new_flipped_boards(self,
                               game_state: TicTacToeGameState,
                               all_boards: torch.Tensor) -> torch.Tensor:
        torch_board = self.learning_data.model.state2tensor(game_state)
        all_boards = self.add_flipped_states(all_boards, torch_board, flip_dims=(2, 3))
        return all_boards

    def add_new_flipped_visits(self, all_visits: torch.Tensor) -> torch.Tensor:
        visits = torch.zeros((1, 3, 3),
                             dtype=torch.float32,
                             device=self.learning_data.device
                             )
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

    def validate(self) -> None:
        win1 = np.array([[1, 1, 1],
                        [-1, 0, -1],
                        [-1, 0, 0]])
        game_state1 = TicTacToeGameState(win1, -1)
        torch_board1 = self.learning_data.model.state2tensor(game_state1)

        draw = np.array([[1, -1, 1],
                        [-1, 1, -1],
                        [-1, 1, -1]])
        game_state2 = TicTacToeGameState(draw, 1)
        torch_board2 = self.learning_data.model.state2tensor(game_state2)

        win2 = np.array([[-1, 1, 1],
                        [-1, 1, -1],
                        [-1, -1, 1]])
        game_state3 = TicTacToeGameState(win2, 1)
        torch_board3 = self.learning_data.model.state2tensor(game_state3)

        with torch.no_grad():
            print('win 1   :', self.learning_data.model(torch_board1))
            print('draw (0):', self.learning_data.model(torch_board2))
            print('win -1  :', self.learning_data.model(torch_board3))


if __name__ == '__main__':
    trainer = Trainer()
    trainer.main()
