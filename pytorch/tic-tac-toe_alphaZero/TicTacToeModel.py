import torch
import torch.nn as nn
from GameRules import TicTacToeGameState


class AlphaZero(nn.Module):
    """Input to forward is a (3, 3, 3) tensor where
    first layer represents player 1, second layer represents player -1,
    and third layer is filled with the value of player to make a move."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device

        self.body_channels = (3, 5, 3)
        self.body_kernels = (3, 3)

        self.out_features1 = 32
        self.out_features2 = 16

        self.policy_channels = (self.body_channels[-1], 3, 1)
        self.policy_kernels = (2, 2)

        self.body = nn.Sequential(
            nn.Conv2d(in_channels=self.body_channels[0],
                      out_channels=self.body_channels[1],
                      kernel_size=self.body_kernels[0],
                      padding=1,
                      device=device),
            nn.ReLU(),
            nn.BatchNorm2d(self.body_channels[1], device=device),
            nn.Conv2d(in_channels=self.body_channels[1],
                      out_channels=self.body_channels[2],
                      kernel_size=self.body_kernels[1],
                      padding=1,
                      device=device),
            nn.ReLU(),
            nn.BatchNorm2d(self.body_channels[2], device=device)
        )

        self.value_head = nn.Sequential(
            nn.Linear(self.body_channels[2]*3*3,
                      self.out_features1, device=device),
            nn.ReLU(),
            nn.Linear(self.out_features1, self.out_features2, device=device),
            nn.ReLU(),
            nn.Linear(self.out_features2, 1, device=device),
            nn.Tanh()
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(in_channels=self.policy_channels[0],
                      out_channels=self.policy_channels[1],
                      kernel_size=self.policy_kernels[0],
                      padding=1,
                      device=device),
            nn.ReLU(),
            nn.BatchNorm2d(self.policy_channels[1], device=device),
            nn.Conv2d(in_channels=self.policy_channels[1],
                      out_channels=self.policy_channels[2],
                      kernel_size=self.policy_kernels[1],
                      device=device),
            nn.ReLU(),
        )

    def forward(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if len(board.shape) == 3:
            board = board.expand((1, -1))

        num_moves = board.shape[0]

        body = self.body(board)
        policies = self.policy_head(body).reshape((num_moves, -1))

        zeros = torch.zeros((1, 9), device=self.device)
        # if policy of a state is identically zero
        if torch.any(torch.all(zeros == policies, dim=1)):
            for i, policy in enumerate(policies):
                if torch.all(zeros == policy):
                    policies[i] = torch.tensor([0.111]*9, dtype=torch.float32,
                                               device=self.device)

        policies = nn.functional.normalize(policies, dim=1, p=1)
        # print(policies)
        # print(board)

        evaluation = self.value_head(
            body.reshape((num_moves, self.body_channels[2]*3*3))
        )
        return evaluation, policies

    def state2tensor(self, game_state: TicTacToeGameState) -> torch.Tensor:
        np_board = torch.from_numpy(game_state.board).to(self.device)
        ones = torch.ones((3, 3)).to(self.device)
        zeros = torch.zeros((3, 3)).to(self.device)

        input = torch.empty((1, 3, 3, 3), device=self.device)
        input[0][0] = (np_board == ones).float()
        input[0][1] = (np_board == -ones).float()
        input[0][2] = ones if game_state.player == 1 else zeros
        return input


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

        self.MSE = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, evaluation: float, result: int,
                policy: torch.Tensor, visits: torch.Tensor) -> torch.Tensor:

        return self.MSE(evaluation, result) + self.cross_entropy(policy, visits)
