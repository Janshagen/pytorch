import random

import numpy as np
import torch
import torch.nn as nn

from AI import MCTSfindMove
from gameplay import availableMoves, gameEnd, makeMove, nextPlayer
from TicTacToeModel import ConvModel, LinearModel

# Constants
LOAD_MODEL = True
SAVE_MODEL = True
FILE = '/home/anton/skola/egen/pytorch/tic-tac-toe/TicTacToeModelConv.pth'

LEARNING_RATE = 0.01
N_EPOCHS = 100_000

OUTPUT_SIZE = 3
HIDDEN_SIZE1 = 72
HIDDEN_SIZE2 = 72

SIMULATIONS = 30
UCB1 = 1.4


def train(model: LinearModel | ConvModel,
          optimizer: torch.optim.Optimizer,
          loss: nn.modules.loss.CrossEntropyLoss,
          device: torch.device) -> None:
    for epoch in range(N_EPOCHS):
        predictions, result = game(model, device)
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


def game(model: LinearModel | ConvModel,
         device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    player = random.choice([1, -1])
    gameState = np.zeros((3, 3), dtype=np.int8)
    input = torch.zeros((1, 3, 3, 3), device=device)
    input[0][2] = torch.ones((3, 3)) * player
    while True:
        move = MCTSfindMove(gameState, player, SIMULATIONS, UCB1)
        makeMove(gameState, player, move)
        player = nextPlayer(player)

        # adding rotated boards
        for i in range(4):
            permutation = model.board2tensor(
                gameState, player, device).rot90(k=i, dims=(2, 3))
            input = torch.cat((input, permutation), dim=0)

        # win
        if gameEnd(gameState).any():
            # adding endboard but with other player to make turn
            input = torch.cat((input, model.board2tensor(
                gameState, -1*player, device)), dim=0)
            predictions = model(input)

            res = [0] if nextPlayer(player) == 1 else [2]
            return predictions, torch.tensor(res, device=device)

        # draw
        if not availableMoves(gameState):
            # adding endboard but with other player to make turn
            input = torch.cat((input, model.board2tensor(
                gameState, -1*player, device)), dim=0)
            predictions = model(input)

            return predictions, torch.tensor([1], device=device)


def validate(model, device):
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
            model(model.board2tensor(win1, -1, device)), 2))
        print('draw:', torch.softmax(
            model(model.board2tensor(draw, 1, device)), 2))
        print('win 2:', torch.softmax(
            model(model.board2tensor(win2, 1, device)), 2))


def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ConvModel(HIDDEN_SIZE1, HIDDEN_SIZE2, OUTPUT_SIZE).to(device)
    if LOAD_MODEL:
        model.load_state_dict(torch.load(FILE))
        model.to(device)
        model.eval()

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    loss = nn.CrossEntropyLoss()

    train(model, optimizer, loss, device)
    validate(model, device)

    if SAVE_MODEL:
        torch.save(model.state_dict(), FILE)


if __name__ == '__main__':
    main()
