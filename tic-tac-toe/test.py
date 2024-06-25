import numpy as np
import torch
import torch.nn as nn
from TicTacToeModel import ConvModel

# x = torch.rand((1, 3), requires_grad=True)
# y = torch.tensor([[1, 2, 2]])
# z = torch.tensor([[1, 4, 6]], dtype=torch.float32, requires_grad=True)
# label = torch.tensor([[1]])
# print(x)

# loss1 = torch.nn.MSELoss()
# l1 = loss1(x[:-1], x[1:])
# l2 = torch.log(torch.abs(label-x[-1]))

# l = l1+l2
# print(l.grad_fn)
# print(l.grad_fn.next_functions)
# print(l.grad_fn.next_functions[0][0].next_functions)
# print(l.grad_fn.next_functions[1][0].next_functions[0][0].next_functions[0]
#       [0].next_functions[1][0].next_functions[0][0].next_functions)


# board = torch.tensor([[1, 0, 1], [0, 1, -1], [-1, 1, -1]], dtype=torch.float32)

# ones = torch.ones((3, 3))
# a = (board == ones).float()
# b = (board == -ones).float()
# input = torch.empty((1, 2, 3, 3))
# input[0][0] = a
# input[0][1] = b

# print(input.shape)

# m = ConvModel(1, 3)

# print(m(input))

# print(torch.softmax(m(input), 2))

# print(torch.softmax(m(input), 2).sum())


# input = torch.randn(3, 5, requires_grad=True)
# target = torch.empty(3, dtype=torch.long).random_(5)

# print(input)
# print(target)

# l = torch.nn.CrossEntropyLoss(reduction='none')
# print(l(input, target))

class ConvModel(nn.Module):
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


model = ConvModel(72, 72, 3)
player = 1
gameState = np.zeros((3, 3))
input = torch.zeros((1, 3, 3, 3))
input[0][2] = torch.ones((3, 3)) * player
gameState[0][1] = 1
player = -1

for i in range(4):
    board = model.board2tensor(
        gameState, player, 'cpu').rot90(k=i, dims=(2, 3))
    input = torch.cat((input, board), dim=0)
