import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from gameplay import availableMoves, gameEnd, makeMove, nextPlayer
from MCTS import AIfindMove

# Constants
SAVE_MODEL = False

FILE = 'tic-tac-toe-model.pth'
LEARNING_RATE = 0.001
N_EPOCHS = 300_000

INPUT_SIZE = 18
OUTPUT_SIZE = 1
HIDDEN_SIZE1 = 36
HIDDEN_SIZE2 = 36

SIMULATIONS = 30
UCB1 = 1.4


class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim1)
        self.hidden = nn.Linear(hidden_dim1, hidden_dim2)
        self.output = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x) -> torch.Tensor:
        out = torch.tanh(self.input(x))
        out = torch.tanh(self.hidden(out))
        out = torch.sigmoid(self.output(out))
        return out

# class Model(nn.Module):
#     def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim) -> None:
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 3, 2, dtype=torch.float64)
#         self.output = nn.Linear(6, 1, dtype=torch.float64)

#     def forward(self, x) -> torch.Tensor:
#         print(x.shape)
#         out = F.relu(self.conv1(x))
#         # print(out)
#         print(out.shape)
#         out = torch.sigmoid(self.output(out))
#         return out
#             input = allGameStates.view(i, 1, 3, 3)


def loss(predictions: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    loss1 = nn.MSELoss()
    l1 = loss1(predictions[:-1], predictions[1:])
    l2 = torch.log(torch.abs(label-predictions[-1]))
    # l = loss1(predictions, label.expand(predictions.shape))
    return l1 + l2


def train(model: nn.Module, optimizer: torch.optim.Optimizer, device):
    # results = []
    # pred = []
    for epoch in range(N_EPOCHS):
        predictions, result = game(model, device)
        # results.append(result.item())
        # pred.append(predictions[-1].item())

        error = loss(predictions, result)
        error.backward()

        optimizer.step()
        optimizer.zero_grad()

        if (N_EPOCHS//1000 + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{N_EPOCHS}], Loss: {error.item():.8f}')


def game(model: nn.Module, device: torch.device) -> torch.Tensor:
    player = random.choice([1, -1])  # X:1  O:-1
    gameState = np.zeros((3, 3))
    stateX = torch.zeros((3, 3), device=device)
    stateO = torch.zeros((3, 3), device=device)
    input = torch.zeros((1, 18), device=device)
    while True:
        move = AIfindMove(gameState, player, SIMULATIONS, UCB1)
        makeMove(gameState, player, move)
        if player == 1:
            makeMove(stateX, 1, move)
        else:
            makeMove(stateO, 1, move)

        input = torch.cat((input, torch.cat((stateX.view(
            1, 9), stateO.view(1, 9)), dim=1)))
        player = nextPlayer(player)

        # win
        if gameEnd(gameState).any():
            predictions = model(input)

            res = 1.00 if nextPlayer(player) == 1 else 0.00
            return predictions, torch.tensor([res], device=device)

        # draw
        if not availableMoves(gameState):
            predictions = model(input)

            return predictions, torch.tensor([0.50], device=device)


def validate(model, device):
    win = torch.tensor([[1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [
                       1, 0, 1], [1, 0, 0]], dtype=torch.float32, device=device).view(1, INPUT_SIZE)
    draw = torch.tensor([[1, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [
                        1, 0, 1], [1, 0, 1]], dtype=torch.float32, device=device).view(1, INPUT_SIZE)

    with torch.no_grad():
        print(model(win).item())
        print(model(draw).item())


def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Model(INPUT_SIZE, HIDDEN_SIZE1,
                  HIDDEN_SIZE2, OUTPUT_SIZE).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    train(model, optimizer, device)
    validate(model, device)

    if SAVE_MODEL:
        torch.save(model.state_dict(), FILE)


if __name__ == '__main__':
    main()
