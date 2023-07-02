import os
import re
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
from GameRules import YatzyGameState


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, device, kernel_size=3, padding=1):
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
    """Input to forward is a (3, 3, 3) tensor where
    first layer represents player 1, second layer represents player -1,
    and third layer is either 1s or 0s."""

    MODEL_PATH = '/home/anton/skola/egen/pytorch/connect4_alphaZero/models/'

    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.body_channels = 5
        self.policy_channels = 3
        self.hidden_nodes = 16

        self.dropout_rate = 0.2

        self.initial_block = nn.Sequential(
            ConvBlock(3, self.body_channels, self.device),
            nn.ReLU()
        )

        self.body = nn.Sequential(
            *[ResidualBlock(self.body_channels,
                            self.body_channels,
                            self.device) for _ in range(6)],
            nn.Dropout2d(self.dropout_rate)
        )

        self.policy_head = nn.Sequential(
            ConvBlock(self.body_channels, self.policy_channels, self.device),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.policy_channels*7*6, 7, device=self.device),
            torch.nn.Softmax(dim=-1),
            nn.Flatten()
        )

        self.value_head = nn.Sequential(
            ConvBlock(self.body_channels, 1, self.device, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7*6, self.hidden_nodes, device=self.device),
            nn.ReLU(),
            nn.Linear(self.hidden_nodes, 1, device=self.device),
            nn.Tanh()
        )

    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if len(boards.shape) == 3:
            boards = torch.unsqueeze(boards, dim=0)

        body = self.initial_block(boards)
        body = self.body(body)

        policies = self.calculate_policies(boards, body)
        evaluation = self.value_head(body)
        return evaluation, policies

    def calculate_policies(self, boards: torch.Tensor,
                           body: torch.Tensor) -> torch.Tensor:
        policies = self.policy_head(body)

        if self.policy_is_identically_zero(policies):
            policies = self.handle_policy_is_zero_case(policies)

        policies = self.set_illegal_moves_to_zero(boards, policies)
        policies = nn.functional.normalize(policies, dim=1, p=1)
        return policies

    def policy_is_identically_zero(self, policies: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros((1, 7), device=self.device)
        return torch.any(torch.all(zeros == policies, dim=1))

    def handle_policy_is_zero_case(self, policies: torch.Tensor) -> torch.Tensor:
        addition = torch.zeros(policies.shape, device=self.device)
        zeros = torch.zeros((1, 7), device=self.device)
        for i, policy in enumerate(policies):
            if torch.all(zeros == policy):
                addition[i] = torch.ones((1, 7), device=self.device)
        policies = policies.clone() + addition
        return policies

    def set_illegal_moves_to_zero(self, boards: torch.Tensor,
                                  policies: torch.Tensor) -> torch.Tensor:
        mask = torch.ones(policies.shape, device=self.device)
        for b, board in enumerate(boards):
            for i in range(7):
                if self.row_is_filled(board, i):
                    mask[b][i] = 0
        masked_policy = policies * mask
        return masked_policy

    def row_is_filled(self, board: torch.Tensor, i: int) -> bool:
        return bool(board[0][0][i] or board[1][0][i])

    def state2tensor(self, game_state: YatzyGameState) -> torch.Tensor:
        np_board = torch.from_numpy(game_state.sheets).to(self.device)
        ones = torch.ones((6, 7)).to(self.device)
        zeros = torch.zeros((6, 7)).to(self.device)

        input = torch.empty((1, 3, 6, 7), device=self.device)
        input[0][0] = (np_board == ones).float()
        input[0][1] = (np_board == -ones).float()
        input[0][2] = ones if game_state.current_player == 1 else zeros
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


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

        self.MSE = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, evaluation: float, result: int,
                policy: torch.Tensor, visits: torch.Tensor) -> torch.Tensor:
        # print(f"eval: {evaluation}")
        # print(f"res: {result}")
        # print(f"policy: {policy}")
        # print(f"visits: {visits}")

        return self.MSE(evaluation, result) + self.cross_entropy(policy, visits)
