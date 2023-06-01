import random

import numpy as np
import torch
import torch.nn as nn
from AI import MCTSfindMove
from gameplay import availableMoves, gameEnd, makeMove, nextPlayer
from MCTSData import MCTSData
from TicTacToeModel import ConvModel

# Constants
LOAD_MODEL = True
SAVE_MODEL = False
FILE = '/home/anton/skola/egen/pytorch/tic-tac-toe/TicTacToeModelConv.pth'

LEARNING_RATE = 0.01
N_EPOCHS = 100_000

OUTPUT_SIZE = 3
HIDDEN_SIZE1 = 72
HIDDEN_SIZE2 = 72

SIMULATIONS = 30
UCB1 = 1.4


def train(data: MCTSData, optimizer: torch.optim.Optimizer,
          loss: nn.modules.loss.CrossEntropyLoss) -> None:
    for epoch in range(N_EPOCHS):
        predictions, result = game(data)
        numMoves = predictions.shape[0]

        error = loss(predictions.reshape(
            (numMoves, 3)), result.expand(numMoves))

        error.backward()

        optimizer.step()
        optimizer.zero_grad()

        if (epoch + 1) % 100 == 0:
            prediction = torch.softmax(predictions.reshape(
                (numMoves, 3))[-1], 0)[int(result.item())].item()
            print(
                f'Epoch [{epoch+1}/{N_EPOCHS}], Loss: {error.item():.8f}',
                f'prediction: {prediction:.4f}, result: {result.item()}')


def game(data: MCTSData) -> tuple[torch.Tensor, torch.Tensor]:
    input = torch.zeros((1, 3, 3, 3), device=data.device)
    input[0][2] = torch.ones((3, 3)) * data.player
    while True:
        move = MCTSfindMove(data)
        makeMove(data.board, data.player, move)
        data.player = nextPlayer(data.player)

        # adding rotated boards
        for i in range(4):
            permutation = data.model.board2tensor(
                data.board, data.player, data.device).rot90(k=i, dims=(2, 3))
            input = torch.cat((input, permutation), dim=0)

        # win
        if gameEnd(data.board):
            # adding endboard but with other player to make turn
            input = torch.cat((input, data.model.board2tensor(
                data.board, -1*data.player, data.device)), dim=0)
            predictions = data.model(input, -1*data.player, data.device)

            res = [0] if nextPlayer(data.player) == 1 else [2]
            return predictions, torch.tensor(res, device=data.device)

        # draw
        if not availableMoves(data.board):
            # adding endboard but with other player to make turn
            input = torch.cat((input, data.model.board2tensor(
                data.board, -1*data.player, data.device)), dim=0)
            predictions = data.model(input, -1*data.player, data.device)

            return predictions, torch.tensor([1], device=data.device)


def validate(data: MCTSData) -> None:
    win1 = np.array([[1, 1, 1],
                     [-1, 0, -1],
                     [-1, 0, 0]])

    draw = np.array([[1, -1, 1],
                     [-1, 1, -1],
                     [-1, 1, -1]])

    win2 = np.array([[-1, 1, 1],
                     [-1, 1, -1],
                     [-1, -1, 1]])

    with torch.no_grad():
        print('win 1:', torch.softmax(
            data.model(win1, -1, data.device), 2))
        print('draw:', torch.softmax(
            data.model(draw, 1, data.device), 2))
        print('win 2:', torch.softmax(
            data.model(win2, 1, data.device), 2))


def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    player = random.choice([1, -1])
    board = np.zeros((3, 3), dtype=np.int8)

    model = ConvModel(HIDDEN_SIZE1, HIDDEN_SIZE2, OUTPUT_SIZE).to(device)
    if LOAD_MODEL:
        model.load_state_dict(torch.load(FILE))
        model.to(device)
        model.eval()

    data = MCTSData(board, player, UCB1, model, device, sim_number=SIMULATIONS)

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    loss = nn.CrossEntropyLoss()

    train(data, optimizer, loss)
    validate(data)

    if SAVE_MODEL:
        torch.save(model.state_dict(), FILE)


if __name__ == '__main__':
    main()
