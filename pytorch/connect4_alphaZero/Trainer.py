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

LEARNING_RATE = 0.1
MOMENTUM = 0.05
WEIGHT_DECAY = 0.01

N_GAMES = 2_500
BATCH_SIZE = 6

SIMULATIONS = 250
EXPLORATION_RATE = 4

LOAD_MODEL_NAME = 'AlphaZero2023-07-28 16:35.pth'
GAMES_FOLDER = '/home/anton/skola/egen/pytorch/connect4_alphaZero/games_5dim/'


class Trainer:
    def __init__(self, load_file: Optional[str] = None):
        self.tt = self.create_training_tools(load_file)
        self.game_simulator = GameSimulator(self.tt.model, EXPLORATION_RATE, SIMULATIONS)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        torch.manual_seed(42)
        self.running_loss = 0.0
        self.running_mse_loss = 0.0

        # [boards, results, visits, game_lengths]
        self.initial_data: list[torch.Tensor]

    def create_training_tools(self, load_file: Optional[str] = None) -> TrainingTools:
        model = self.create_model(load_file)
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=LEARNING_RATE,
                                    momentum=MOMENTUM,
                                    weight_decay=WEIGHT_DECAY)
        loss = Loss()
        return TrainingTools(model, loss, optimizer, N_GAMES)

    def create_model(self, load_file: Optional[str] = None) -> AlphaZero:
        model = AlphaZero()
        if LOAD_MODEL:
            model = model.load_model(load_file)
        model.train()
        return model

    def train_and_validate(self) -> None:
        print("Training Started")
        self.tt.visualizer.visualize_model(self.tt.model)
        self.train()
        self.validate()
        self.tt.visualizer.close()

        if SAVE_MODEL:
            self.tt.save_model()

    def train(self) -> None:
        for game in range(N_GAMES):
            boards, results, visits = \
                self.game_simulator.create_N_data_points(number_games=1)

            permutation = torch.randperm(boards.shape[0])
            boards = boards[permutation]
            results = results[permutation]
            visits = visits[permutation]

            evaluations = total_error = torch.tensor([])
            for i in range(0, boards.shape[0], BATCH_SIZE):
                evaluations, total_error = self.update_weights(
                    boards[i:i + BATCH_SIZE],
                    results[i:i + BATCH_SIZE],
                    visits[i:i + BATCH_SIZE])

            self.tt.scheduler.step()
            self.print_info(game, evaluations, results, total_error)
            self.save_model(game)
            self.write_loss(game)

    def update_weights(self, boards: torch.Tensor,
                       results: torch.Tensor,
                       visits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.tt.optimizer.zero_grad()
        evaluations, policies = self.get_predictions(boards)

        total_error, mse_error = self.tt.loss.forward(
            evaluations, results, policies, visits
        )

        total_error.backward()
        self.tt.optimizer.step()

        self.running_loss += total_error.item()
        self.running_mse_loss += mse_error.item()

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
        if (batch+1) % (N_GAMES/10) == 0:
            print(
                f'Batch [{batch+1}/{N_GAMES}], Loss: {error.item():.8f},',
                f'evaluation: {evaluations[-1].item():.4f},',
                f'result: {result[0][0].item()}')

    def write_loss(self, batch: int) -> None:
        if (batch+1) % (N_GAMES/100) == 0:
            self.tt.visualizer.add_loss("Total Loss",
                                        self.running_loss/N_GAMES*100,
                                        (batch+1)*BATCH_SIZE)
            self.tt.visualizer.add_loss("MSE Loss",
                                        self.running_mse_loss/N_GAMES*100,
                                        (batch+1)*BATCH_SIZE)
            self.tt.visualizer.add_loss("Cross Entropy Loss",
                                        (self.running_loss-self.running_mse_loss) /
                                        N_GAMES*100,
                                        (batch+1)*BATCH_SIZE)
            self.running_loss = 0.0
            self.running_mse_loss = 0.0

    def save_model(self, batch: int) -> None:
        if (batch+1) % (N_GAMES/10) == 0 and SAVE_MODEL:
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
