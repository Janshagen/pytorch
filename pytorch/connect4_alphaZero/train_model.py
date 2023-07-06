import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from TrainingData import TrainingData
from GameRules import Connect4GameState
from MCTS import MCTS
from Connect4Model import AlphaZero, Loss

# Constants
LOAD_MODEL = False
SAVE_MODEL = True

LEARNING_RATE = 0.2
MOMENTUM = 0.9
WEIGHT_DECAY = 0.01

N_BATCHES = 3_000
BATCH_SIZE = 5

SIMULATIONS = 50
EXPLORATION_RATE = 3


class Trainer:
    def __init__(self, load_file: Optional[str] = None):
        self.learning_data = self.create_learning_data(load_file)
        self.mcts = MCTS(self.learning_data.model,
                         EXPLORATION_RATE,
                         sim_number=SIMULATIONS
                         )

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
                                    momentum=MOMENTUM,
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
            for sample in range(BATCH_SIZE):
                game_boards, game_result, game_visits = self.get_game_data()

                boards = torch.cat((boards, game_boards), dim=0)
                results = torch.cat((results, game_result), dim=0)
                visits = torch.cat((visits, game_visits), dim=0)

            evaluations, policies = self.learning_data.model(boards)

            error = self.learning_data.loss(evaluations, results, policies, visits)
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

    def print_info(self, batch: int, evaluations: torch.Tensor,
                   result: torch.Tensor, error: torch.Tensor) -> None:
        print(
            f'Batch [{batch+1}/{N_BATCHES}], Loss: {error.item():.8f},',
            f'evaluation: {evaluations[-1].item():.4f}, result: {result[0][0].item()}')

    def game(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        game_state = Connect4GameState.new_game(random.choice([1, -1]))
        all_boards = self.new_boards(game_state)

        all_visits = torch.tensor([], device=self.learning_data.device)
        while True:
            game_state = self.find_and_make_move(game_state)
            all_boards = self.add_new_flipped_boards(game_state, all_boards)
            all_visits = self.add_new_flipped_visits(all_visits)

            if game_state.game_over():
                visits = torch.tensor([0.142857143]*7, dtype=torch.float32,
                                      device=self.learning_data.device)

                all_visits = self.add_flipped_states(all_visits,
                                                     visits.unsqueeze(dim=0),
                                                     flip_dims=(1,)
                                                     )

                result = torch.tensor([game_state.get_status()], dtype=torch.float32,
                                      device=self.learning_data.device)
                return all_boards, result, all_visits

    def new_boards(self, game_state: Connect4GameState) -> torch.Tensor:
        boards = torch.zeros((2, 4, 6, 7), device=self.learning_data.device)
        for i in range(2):
            if game_state.player == 1:
                boards[i][2] = torch.ones((6, 7))
            elif game_state.player == -1:
                boards[i][3] = torch.ones((6, 7))
        return boards

    def find_and_make_move(self, game_state: Connect4GameState) -> Connect4GameState:
        move = self.mcts.find_move(game_state)
        game_state.make_move(move)
        return game_state

    def add_new_flipped_boards(self,
                               game_state: Connect4GameState,
                               all_boards: torch.Tensor) -> torch.Tensor:
        torch_board = self.learning_data.model.state2tensor(game_state)
        all_boards = self.add_flipped_states(all_boards, torch_board, flip_dims=(3,))
        return all_boards

    def add_new_flipped_visits(self, all_visits: torch.Tensor) -> torch.Tensor:
        visits = [0]*7
        for child in self.mcts.root.children:
            visits[child.move] = child.visits

        visits = torch.tensor(visits,
                              dtype=torch.float32,
                              device=self.learning_data.device
                              )
        visits = nn.functional.normalize(visits, dim=0, p=1)
        visits = torch.unsqueeze(visits, dim=0)

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

    def validate(self) -> None:
        win1 = np.array([[0,  1,  1, -1, 0,  1,  1],
                        [0, -1, -1, -1,  0, -1, -1],
                        [1,  1,  1, -1,  0,  1,  1],
                        [-1, -1,  1,  1,  0,  1, -1],
                        [1,  1, -1, -1,  1, -1, 1],
                        [-1, -1,  1, -1, -1,  1, -1]])
        game_state1 = Connect4GameState(win1, -1)
        torch_board1 = self.learning_data.model.state2tensor(game_state1)

        draw = np.array([[1,  1, -1,  1, -1, -1, 1],
                        [-1,  -1,  1,  1, -1,  1, -1],
                        [1, -1,  1, -1, -1, -1,  1],
                        [-1,  1,  1, -1,  1, -1,  1],
                        [-1, -1, -1,  1, -1,  1, -1],
                        [1,  1,  1, -1,  1, -1, 1]])
        game_state2 = Connect4GameState(draw, -1)
        torch_board2 = self.learning_data.model.state2tensor(game_state2)

        win2 = np.array([[0,  0,  0,  0,  0,  0,  0],
                        [0, -1, -1, -1, -1,  1,  0],
                        [-1,  1,  1,  1, -1, 1, 0],
                        [1, -1, -1,  1,  1, -1, 0],
                        [-1,  1,  1, -1, 1, -1,  1],
                        [1, -1,  1, -1, -1, -1,  1]])
        game_state3 = Connect4GameState(win2, 1)
        torch_board3 = self.learning_data.model.state2tensor(game_state3)

        with torch.no_grad():
            print('win 1   :', self.learning_data.model(torch_board1))
            print('draw (0):', self.learning_data.model(torch_board2))
            print('win -1  :', self.learning_data.model(torch_board3))


if __name__ == '__main__':
    trainer = Trainer()
    trainer.main()
