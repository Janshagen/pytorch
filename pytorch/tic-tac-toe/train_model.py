import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from gameplay import availableMoves, gameEnd, makeMove, nextPlayer
from MCTS import AIfindMove
from TicTacToeModel import ConvModel

# Constants
SAVE_MODEL = True
FILE = '/home/anton/skola/egen/pytorch/tic-tac-toe/TicTacToeModelConv.pth'

LEARNING_RATE = 0.01
N_EPOCHS = 100_000

OUTPUT_SIZE = 3
HIDDEN_SIZE1 = 64
HIDDEN_SIZE2 = 64

SIMULATIONS = 30
UCB1 = 1.4


def train(model: nn.Module, optimizer: torch.optim.Optimizer, loss, device: torch.device):
    for epoch in range(N_EPOCHS):
        predictions, result = game(model, device)
        numMoves = predictions.shape[0]

        error = loss(predictions.reshape(
            (numMoves, 3)), result.expand(numMoves))

        error.backward()

        optimizer.step()
        optimizer.zero_grad()

        if (epoch + 1) % 100 == 0:
            print(
                f'Epoch [{epoch+1}/{N_EPOCHS}], Loss: {error.item():.8f}, predictions: {torch.softmax(predictions.reshape((numMoves, 3))[-1], 0)[result.item()].item():.4f}, result: {result.item()}')


def game(model: nn.Module, device: torch.device) -> torch.Tensor:
    player = random.choice([1, -1])
    gameState = np.zeros((3, 3))
    input = torch.zeros((1, 2, 3, 3), device=device)
    while True:
        move = AIfindMove(gameState, player, SIMULATIONS, UCB1)
        makeMove(gameState, player, move)

        input = torch.cat(
            (input, model.board2tensor(gameState, device)), dim=0)
        player = nextPlayer(player)

        # win
        if gameEnd(gameState).any():
            predictions = model(input)

            res = [0] if nextPlayer(
                player) == 1 else [2]
            return predictions, torch.tensor(res, device=device)

        # draw
        if not availableMoves(gameState):
            predictions = model(input)

            return predictions, torch.tensor([1], device=device)


def validate(model, device):
    win = np.array([[1, 1, 1], [-1, 0, -1], [-1, 0, 0]])
    draw = np.array([[1, -1, 1], [-1, 1, -1], [-1, 1, -1]])

    with torch.no_grad():
        print(torch.softmax(model(model.board2tensor(win, device)), 2))
        print(torch.softmax(model(model.board2tensor(draw, device)), 2))


def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ConvModel(HIDDEN_SIZE1, HIDDEN_SIZE2, OUTPUT_SIZE).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    loss = nn.CrossEntropyLoss()

    train(model, optimizer, loss, device)
    validate(model, device)

    if SAVE_MODEL:
        torch.save(model.state_dict(), FILE)


if __name__ == '__main__':
    main()
