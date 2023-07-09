import os
import re
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
from GameRules import TicTacToeGameState


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
    """Input to forward is a (4, 3, 3) tensor where
    first layer represents player 1, second layer represents player -1,
    and third and fourth layers are either 1s or 0s depending on current player."""

    MODEL_PATH = '/home/anton/skola/egen/pytorch/tic-tac-toe_alphaZero/models/'

    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        alpha = 1
        self.dirichlet = torch.distributions.dirichlet.Dirichlet(
            torch.ones((1, 9))*alpha
        )

        self.body_channels = 5
        self.policy_channels = 4
        self.hidden_nodes = 32

        self.dropout_rate = 0.2

        self.initial_block = nn.Sequential(
            ConvBlock(4, self.body_channels, self.device),
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
            nn.Conv2d(self.policy_channels, 1,
                      kernel_size=3,
                      padding=1,
                      device=self.device),
            nn.BatchNorm2d(1, device=self.device),
            torch.nn.Softmax(dim=-1)
        )

        self.value_head = nn.Sequential(
            ConvBlock(self.body_channels, 1, self.device, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3*3, self.hidden_nodes, device=self.device),
            nn.ReLU(),
            nn.Linear(self.hidden_nodes, 1, device=self.device),
            nn.Tanh()
        )

    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        body = self.initial_block(boards)
        body = self.body(body)

        policies = self.calculate_policies(boards, body)
        evaluation = self.value_head(body)
        return evaluation, policies

    def calculate_policies(self, boards: torch.Tensor,
                           body: torch.Tensor) -> torch.Tensor:
        policies = self.policy_head(body)

        # if self.policy_is_identically_zero(policies):
        #     policies = self.handle_policy_is_zero_case(policies)

        policies = self.set_illegal_moves_to_zero(boards, policies)
        policies = policies.reshape(-1, 9)
        policies = nn.functional.normalize(policies, dim=1, p=1)
        return policies

    def policy_is_identically_zero(self, policies: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros((1, 9), device=self.device)
        return torch.any(torch.all(zeros == policies, dim=1))

    def handle_policy_is_zero_case(self, policies: torch.Tensor) -> torch.Tensor:
        addition = torch.zeros(policies.shape, device=self.device)
        zeros = torch.zeros((1, 9), device=self.device)
        for i, policy in enumerate(policies):
            if torch.all(zeros == policy):
                addition[i] = torch.ones((1, 9), device=self.device)
        policies = policies.clone() + addition
        return policies

    def set_illegal_moves_to_zero(self, boards: torch.Tensor,
                                  policies: torch.Tensor) -> torch.Tensor:
        ones = torch.ones(policies.shape[1:], device=self.device)
        mask = torch.ones(policies.shape, device=self.device)

        player_one = boards[:, 0] == ones
        player_two = boards[:, 1] == ones
        illegal_indices = (player_one + player_two).unsqueeze(dim=1)

        mask[illegal_indices] = 0
        return policies * mask

    def add_noise(self, policy: torch.Tensor):
        sample_size = torch.Size((policy.shape[0],))
        noise = self.dirichlet.sample(sample_size)[0].to(self.device)
        return 0.75*policy + 0.25*noise

    def state2tensor(self, game_state: TicTacToeGameState) -> torch.Tensor:
        np_board = torch.from_numpy(game_state.board).to(self.device)
        ones = torch.ones((3, 3)).to(self.device)
        zeros = torch.zeros((3, 3)).to(self.device)

        input = torch.empty((1, 4, 3, 3), device=self.device)
        input[0][0] = (np_board == ones).float()
        input[0][1] = (np_board == -ones).float()
        input[0][2] = ones if game_state.player == 1 else zeros
        input[0][3] = ones if game_state.player == -1 else zeros
        return input

    def load_model(self, file: Optional[str] = None):
        model = AlphaZero()
        model.load_state_dict(torch.load(self.get_load_file(file)))
        model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def get_load_file(file: Optional[str] = None) -> str:
        if file:
            return AlphaZero.MODEL_PATH + file

        model_name = re.compile(".*?AlphaZero(.*?).pth")
        files = os.listdir(AlphaZero.MODEL_PATH)

        newest_file_datetime = datetime(2000, 1, 1, 1, 1)
        newest_index = 0
        for i, file in enumerate(files):
            matches = model_name.match(file)
            if not matches:
                continue

            file_datetime = datetime.strptime(matches.group(1), '%Y-%m-%d %H:%M')
            if file_datetime > newest_file_datetime:
                newest_file_datetime = file_datetime
                newest_index = i

        return AlphaZero.MODEL_PATH + files[newest_index]


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

        self.MSE = nn.MSELoss(reduction="none")
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, evaluation: torch.Tensor, result: torch.Tensor,
                policy: torch.Tensor, visits: torch.Tensor,
                game_lengths: list[int]) -> torch.Tensor:

        cross_entropy = self.cross_entropy.forward(policy, visits)

        mse = self.MSE.forward(evaluation, result)
        mse_weight = torch.tensor([], device=mse.device)
        for length in game_lengths:
            new_weight = torch.linspace(
                start=0, end=1, steps=length,
                device=mse.device).unsqueeze(dim=1)
            new_weight = new_weight*new_weight.sum().item()
            mse_weight = torch.cat((mse_weight, new_weight))

        weighted_mse = torch.mean(mse*mse_weight)

        return 10 * weighted_mse + cross_entropy
