import os
import re
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
from GameRules import Connect4GameState


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 device: torch.device, kernel_size: int = 2, padding: int = 1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=padding,
                      device=device),
            nn.BatchNorm2d(out_channels, device=device),
        )

    def forward(self, X):
        return self.model(X)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, device, kernel_size=3, padding=1):
        super().__init__()

        self.model = nn.Sequential(
            ConvBlock(in_channels, out_channels, device, kernel_size, padding),
            nn.ReLU(),
            ConvBlock(in_channels, out_channels, device, kernel_size, padding)
        )
        self.relu = nn.ReLU()

    def forward(self, X):
        Y = self.model(X)
        Y = Y + X
        return self.relu(Y)


class AlphaZero(nn.Module):
    """Input to forward is a concatenation of n (5, 6, 7) tensors where
    first layer represents empty cells, second represents player 1,
    second layer represents player -1,
    and third and fourth layers are either 1s or 0s depending on current player."""

    MODEL_PATH = '/home/anton/skola/egen/pytorch/connect4_alphaZero/models/'

    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.noise_ratio = 0.25
        alpha = 1
        self.dirichlet = torch.distributions.dirichlet.Dirichlet(
            torch.ones((1, 7))*alpha
        )

        self.body_channels = 2
        self.policy_channels = 1
        self.value_channels = 1

        self.number_of_residual_blocks = 1
        self.dropout_rate = 0.2

        self.initial_block = nn.Sequential(
            ConvBlock(3, self.body_channels, self.device),
            nn.ReLU(),
            ConvBlock(self.body_channels, self.body_channels, self.device, padding=0),
            nn.ReLU()
        )

        self.body = nn.Sequential(
            *[ResidualBlock(self.body_channels,
                            self.body_channels,
                            self.device) for _ in range(self.number_of_residual_blocks)],
            nn.Dropout2d(self.dropout_rate)
        )

        self.policy_head = nn.Sequential(
            ConvBlock(self.body_channels, self.policy_channels, self.device,
                      kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.policy_channels*7*6, 7, device=self.device),
            nn.Softmax(dim=-1)
        )

        self.value_head = nn.Sequential(
            ConvBlock(self.body_channels, self.value_channels, self.device,
                      kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.value_channels*7*6, 1, device=self.device)
        )

    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if len(boards.shape) == 3:
            boards = torch.unsqueeze(boards, dim=0)

        body = self.initial_block.forward(boards)
        body = self.body.forward(body)

        policies = self.policy_head.forward(body)

        evaluation = self.value_head.forward(body)
        evaluation = torch.tanh(0.2*evaluation)
        return evaluation, policies

    def add_noise(self, policy: torch.Tensor):
        noise = self.dirichlet.sample(torch.Size((1,)))[0].to(self.device)
        return (1-self.noise_ratio) * policy + self.noise_ratio * noise

    def state2tensor(self, game_state: Connect4GameState) -> torch.Tensor:
        np_board = torch.from_numpy(game_state.board).to(self.device)
        ones = torch.ones((6, 7)).to(self.device)
        zeros = torch.zeros((6, 7)).to(self.device)

        # ändra till två (eller tre) lager
        input = torch.empty((1, 3, 6, 7), device=self.device)
        current_player = game_state.player
        input[0][0] = (np_board == current_player*ones).float()
        input[0][1] = (np_board == -current_player*ones).float()
        input[0][2] = (np_board == zeros).float()
        return input

    def load_model(self, file: Optional[str] = None):
        model = AlphaZero()
        model.load_state_dict(torch.load(self.get_load_file(file)))
        model.to(self.device)
        return model

    @staticmethod
    def get_load_file(file: Optional[str] = None) -> str:
        if file:
            return AlphaZero.MODEL_PATH + file

        model_name = re.compile(".*?AlphaZero(.*?).pth")
        files = os.listdir(AlphaZero.MODEL_PATH)

        oldest_file_datetime = datetime(2000, 1, 1, 1, 1)
        oldest_index = 0
        for i, file in enumerate(files):
            matches = model_name.match(file)
            if not matches:
                continue

            file_datetime = datetime.strptime(matches.group(1), '%Y-%m-%d %H:%M')
            if file_datetime > oldest_file_datetime:
                oldest_file_datetime = file_datetime
                oldest_index = i

        return AlphaZero.MODEL_PATH + files[oldest_index]

    @staticmethod
    def reshape_and_normalize(policy: torch.Tensor) -> torch.Tensor:
        policy = policy.reshape((-1, 7))
        policy = torch.nn.functional.normalize(policy, p=1, dim=1)
        return policy


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

        self.MSE = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, evaluation: torch.Tensor, result: torch.Tensor,
                policy: torch.Tensor, visits: torch.Tensor,
                game_lengths: Optional[torch.Tensor] = None) \
            -> tuple[torch.Tensor, torch.Tensor]:

        cross_entropy = self.cross_entropy.forward(policy, visits)

        mse = self.MSE.forward(evaluation, result)
        return mse + cross_entropy, mse
