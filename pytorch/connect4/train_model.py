from crypt import methods
import random

import numpy as np
import torch
import torch.nn as nn

from AI import MCTSfindMove
from Connect4Model import Model
from gameplay import availableMoves, gameEnd, makeMove, nextPlayer

# Constants
# Saving
LOAD_MODEL = True
SAVE_MODEL = False
FILE = '/home/anton/skola/egen/pytorch/connect4/Connect4model.pth'

# Learning
LEARNING_RATE = 0.01
N_EPOCHS = 100_000

# Model architecture
OUT_CHANNELS1 = 6
OUT_CHANNELS2 = 6
HIDDEN_SIZE1 = 120
HIDDEN_SIZE2 = 72

# MCTS
SIMULATIONS = 50
UCB1 = 1.4


def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    model = Model(OUT_CHANNELS1, OUT_CHANNELS2,
                  HIDDEN_SIZE1, HIDDEN_SIZE2).to(device)
    if LOAD_MODEL:
        model.load_state_dict(torch.load(FILE, map_location='cpu'))
        model.to(device)
        model.eval()

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    loss = nn.CrossEntropyLoss()

    # train(model, optimizer, loss, device)
    validate(model, device)

    if SAVE_MODEL:
        torch.save(model.state_dict(), FILE)


def train(model: nn.Module, optimizer: torch.optim.Optimizer, loss: nn.CrossEntropyLoss, device: torch.device) -> None:
    for epoch in range(N_EPOCHS):
        predictions, result = training_game(model, device)
        numMoves = predictions.shape[0]

        error = loss(predictions.reshape(
            (numMoves, 3)), result.expand(numMoves))

        error.backward()

        optimizer.step()
        optimizer.zero_grad()

        if (epoch + 1) % 100 == 0:
            print(
                f'Epoch [{epoch+1}/{N_EPOCHS}], Loss: {error.item():.4f}, '
                f'prediction: {torch.softmax(predictions.reshape((numMoves, 3))[-1], 0)[result.item()].item():.4f}, '
                f'result: {result.item()}')

        if (epoch+1) % 1000 == 0 and SAVE_MODEL:
            torch.save(model.state_dict(), FILE)


def training_game(model: nn.Module, device: torch.device) -> 'tuple[torch.Tensor]':
    player = random.choice([1, -1])
    gameState = np.zeros((6, 7))
    game_history = torch.zeros((1, 3, 6, 7), device=device)
    game_history[0][2] = torch.ones((6, 7)) * player
    while True:
        move = MCTSfindMove(gameState, player, SIMULATIONS, UCB1)
        row = makeMove(gameState, player, move)
        player = nextPlayer(player)

        # adding flipped boards to history
        game_history = add_to_history(game_history, gameState, player,
                                      model.board2tensor, device)

        # win
        if gameEnd(gameState, row, move).any():
            # adding endboard but with other player to make turn
            game_history = add_to_history(game_history, gameState, -player,
                                          model.board2tensor, device)
            predictions = model(game_history)

            res = [0] if nextPlayer(player) == 1 else [2]
            return predictions, torch.tensor(res, device=device)

        # draw
        if not availableMoves(gameState):
            # adding endboard but with other player to make turn
            game_history = add_to_history(game_history, gameState, -player,
                                          model.board2tensor, device)
            predictions = model(game_history)

            return predictions, torch.tensor([1], device=device)


def add_to_history(game_history: torch.Tensor, gameState: np.ndarray, player: int, board2tensor, device: torch.device) -> None:
    original = board2tensor(gameState, player, device)
    flipped = board2tensor(np.flip(gameState, 1).copy(), player, device)
    game_history = torch.cat((game_history, original), dim=0)
    game_history = torch.cat((game_history, flipped), dim=0)
    return game_history


def validate(model: nn.Module, device: torch.device) -> None:
    def game():
        player = 1
        gameState = np.zeros((1, 6, 7))
        game_history = np.zeros((1, 6, 7))
        while True:
            move = MCTSfindMove(gameState[0], player, SIMULATIONS, UCB1)
            row = makeMove(gameState[0], player, move)
            player = nextPlayer(player)

            game_history = np.concatenate((game_history, gameState))

            # win
            if gameEnd(gameState[0], row, move).any() or not availableMoves(gameState[0]):
                return game_history

    with torch.no_grad():
        history = game()
        player = 1
        for gameState in history:
            print(gameState, '\n Predictions:',
                  torch.softmax(model(model.board2tensor(gameState, player, device)), dim=2), '\n')
            player = nextPlayer(player)


if __name__ == '__main__':
    main()
