import torch
import torch.nn as nn
import numpy as np


class LinearModel(nn.Module):
    """Output is probability of player 1 to win. 
    Input to forward is a (1, 18) tensor where first 9 values 
    represents player 1 positions and second layer represents 
    player -1 positions."""

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim1)
        self.hidden1 = nn.Linear(hidden_dim1, hidden_dim2)
        self.hidden2 = nn.Linear(hidden_dim2, hidden_dim3)
        self.output = nn.Linear(hidden_dim3, output_dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        x = torch.relu(self.input(x))
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.sigmoid(self.output(x))
        return x

    def board2tensor(self, board: np.array, device: torch.device) -> torch.Tensor:
        board = torch.from_numpy(board).to(device)
        ones = torch.ones((3, 3)).to(device)
        a = (board == ones).float().reshape(1, 9)
        b = (board == -ones).float().reshape(1, 9)
        return torch.cat((a, b), dim=1)


class ConvModel(nn.Module):
    """Softmax of output is probability of winning or drawing as 
    [win(1), draw, win(-1)]. Input to froward is a (3, 3, 3) tensor where 
    first layer represents player 1, second layer represents player -1, 
    and third layer is filled with the value of player to make a move."""

    def __init__(self, hidden_dim1, hidden_dim2, output_dim) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 2)
        self.linear1 = nn.Linear(12, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.linear3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = torch.relu(self.linear1(x.reshape(-1, 1, 12)))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def board2tensor(self, board: np.array, player: int, device: torch.device) -> torch.Tensor:
        board = torch.from_numpy(board).to(device)
        ones = torch.ones((3, 3)).to(device)
        a = (board == ones).float()
        b = (board == -ones).float()

        input = torch.empty((1, 3, 3, 3), device=device)
        input[0][0] = a
        input[0][1] = b
        input[0][2] = player*ones
        return input
