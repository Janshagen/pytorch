from typing import Optional

import numpy as np
import torch
from TrainingTools import TrainingTools
from GameRules import Connect4GameState
from Connect4Model import AlphaZero, Loss
from GameSimulator import GameSimulator


# Constants
LOAD_MODEL = False
SAVE_MODEL = True

LEARNING_RATE = 0.2
MOMENTUM = 0.9
WEIGHT_DECAY = 0.01

N_BATCHES = 2_500
BATCH_SIZE = 1

SIMULATIONS = 10
EXPLORATION_RATE = 3

LOAD_MODEL_NAME = 'AlphaZero2023-07-28 16:35.pth'
GAMES_FOLDER = '/home/anton/skola/egen/pytorch/connect4_alphaZero/games_5dim/'


class Trainer:
    def __init__(self, load_file: Optional[str] = None):
        self.tt = self.create_training_tools(load_file)
        self.game_simulator = GameSimulator(self.tt.model, EXPLORATION_RATE, SIMULATIONS)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.running_loss = 0.0
        self.running_mse_loss = 0.0

        # [boards, results, visits, game_lengths]
        self.initial_data: list[torch.Tensor]
        self.second_data: list[torch.Tensor]

    def train_and_validate(self) -> None:
        print("Training Started")
        self.tt.visualizer.visualize_model(self.tt.model)
        self.initial_data = self.load_data(range(5))
        self.second_data = self.load_data(range(5, 10))
        self.train()
        self.validate()
        self.tt.visualizer.close()

        if SAVE_MODEL:
            self.tt.save_model()

    def create_training_tools(self, load_file: Optional[str] = None) -> TrainingTools:
        model = self.create_model(load_file)
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=LEARNING_RATE,
                                    momentum=MOMENTUM,
                                    weight_decay=WEIGHT_DECAY)
        loss = Loss()
        return TrainingTools(model, loss, optimizer, N_BATCHES)

    def create_model(self, load_file: Optional[str] = None) -> AlphaZero:
        model = AlphaZero()
        if LOAD_MODEL:
            model = model.load_model(load_file)
        model.train()
        return model

    def train(self) -> None:
        self.update_weights(*self.initial_data[:3])
        self.update_weights(*self.second_data[:3])

        for batch in range(2, N_BATCHES):
            boards, results, visits, _ = \
                self.game_simulator.create_N_data_points(BATCH_SIZE)

            evaluations, total_error = self.update_weights(boards, results, visits)

            self.print_info(batch, evaluations, results, total_error)
            self.save_model(batch)
            self.write_loss(batch)

    def update_weights(self, boards: torch.Tensor,
                       results: torch.Tensor,
                       visits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.tt.optimizer.zero_grad()
        evaluations, policies = self.get_predictions(boards)

        total_error, mse_error = self.tt.loss.forward(
            evaluations, results, policies, visits
        )

        total_error.backward()
        self.running_loss += total_error.item()
        self.running_mse_loss += mse_error.item()

        self.tt.optimizer.step()
        self.tt.scheduler.step()

        return evaluations, total_error

    def get_predictions(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        evaluations, policies = self.tt.model.forward(boards)
        policies = self.mask_illegal_moves(boards, policies)
        policies = AlphaZero.reshape_and_normalize(policies)
        return evaluations, policies

    def mask_illegal_moves(self, boards: torch.Tensor,
                           policies: torch.Tensor) -> torch.Tensor:
        masks = Connect4GameState.get_masks(boards)
        return policies * masks

    def print_info(self, batch: int, evaluations: torch.Tensor,
                   result: torch.Tensor, error: torch.Tensor) -> None:
        # if (batch+1) % (N_BATCHES/10) == 0:
        print(
            f'Batch [{batch+1}/{N_BATCHES}], Loss: {error.item():.8f},',
            f'evaluation: {evaluations[-1].item():.4f},',
            f'result: {result[0][0].item()}')

    def write_loss(self, batch: int) -> None:
        if (batch+1) % (N_BATCHES/100) == 0:
            self.tt.visualizer.add_loss("Total Loss",
                                        self.running_loss/N_BATCHES*100,
                                        (batch+1)*BATCH_SIZE)
            self.tt.visualizer.add_loss("MSE Loss",
                                        self.running_mse_loss/N_BATCHES*100,
                                        (batch+1)*BATCH_SIZE)
            self.tt.visualizer.add_loss("Cross Entropy Loss",
                                        (self.running_loss-self.running_mse_loss) /
                                        N_BATCHES*100,
                                        (batch+1)*BATCH_SIZE)
            self.running_loss = 0.0
            self.running_mse_loss = 0.0

    def save_model(self, batch: int) -> None:
        if (batch+1) % (N_BATCHES/10) == 0 and SAVE_MODEL:
            self.tt.save_model()

    def load_data(self, file_numbers: range) -> list[torch.Tensor]:
        data = [torch.tensor([], device=self.device) for _ in range(4)]
        for i in file_numbers:
            with open(GAMES_FOLDER + f"game_{i}.pt", 'rb') as file:
                new_data = torch.load(file)
                for j in range(4):
                    data[j] = torch.cat((data[j], new_data[j]))
        return data

    def validate(self) -> None:
        win1 = np.array([[0,  1,  1, -1, 0,  1,  1],
                        [0, -1, -1, -1,  0, -1, -1],
                        [1,  1,  1, -1,  0,  1,  1],
                        [-1, -1,  1,  1,  0,  1, -1],
                        [1,  1, -1, -1,  1, -1, 1],
                        [-1, -1,  1, -1, -1,  1, -1]])
        game_state1 = Connect4GameState(win1, -1)
        torch_board1 = self.tt.model.state2tensor(game_state1)

        draw = np.array([[1,  1, -1,  1, -1, -1, 1],
                        [-1,  -1,  1,  1, -1,  1, -1],
                        [1, -1,  1, -1, -1, -1,  1],
                        [-1,  1,  1, -1,  1, -1,  1],
                        [-1, -1, -1,  1, -1,  1, -1],
                        [1,  1,  1, -1,  1, -1, 1]])
        game_state2 = Connect4GameState(draw, -1)
        torch_board2 = self.tt.model.state2tensor(game_state2)

        win2 = np.array([[0,  0,  0,  0,  0,  0,  0],
                        [0, -1, -1, -1, -1,  1,  0],
                        [-1,  1,  1,  1, -1, 1, 0],
                        [1, -1, -1,  1,  1, -1, 0],
                        [-1,  1,  1, -1, 1, -1,  1],
                        [1, -1,  1, -1, -1, -1,  1]])
        game_state3 = Connect4GameState(win2, 1)
        torch_board3 = self.tt.model.state2tensor(game_state3)

        with torch.no_grad():
            print('win 1   :', self.tt.model(torch_board1))
            print('draw (0):', self.tt.model(torch_board2))
            print('win -1  :', self.tt.model(torch_board3))


if __name__ == '__main__':
    trainer = Trainer(LOAD_MODEL_NAME)
    trainer.train_and_validate()
