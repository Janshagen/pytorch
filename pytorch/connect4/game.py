import random

import numpy as np
import pygame
import torch

from AI import MCTSfindMove, loadModel
from gameplay import availableMoves, gameEnd, makeMove, nextPlayer
from interface import (draw, gameOver, initializeGame,
                       resolveEvent)

# Configurations
SIMULATIONS = 1000
WIDTH = 120
HEIGHT = WIDTH*0.8
UCB1 = 1.4

FILE = '/home/anton/skola/egen/pytorch/connect4/Connect4model10V1.pth'


def game(model: torch.nn.Module, device: torch.device) -> tuple:
    player = random.choice([1, -1])
    gameState, screen, frame = initializeGame(WIDTH, HEIGHT)
    while True:
        if player == 1:
            # Human
            move = resolveEvent(gameState, player, WIDTH)
            row = makeMove(gameState, player, move)
            if type(move) == int:
                player = nextPlayer(player)

        elif player == -1:
            # Neural network
            print('Prediction after human move:', torch.softmax(model(model.board2tensor(
                gameState, player, device))[0][0], dim=0))
            move = MCTSfindMove(gameState, player, SIMULATIONS,
                                UCB1, model, device, cutoff=True)
            row = makeMove(gameState, player, move)
            player = nextPlayer(player)
            resolveEvent(gameState, 0, WIDTH)
            print(f'Prediction after AIs move:', torch.softmax(model(model.board2tensor(
                gameState, player, device))[0][0], dim=0))

        draw(screen, frame, gameState, WIDTH, HEIGHT, move, player)
        if gameEnd(gameState, row, move).any():
            return (row, move)

        if not availableMoves(gameState):
            return (row, move)


def validationGame(model: torch.nn.Module, device: torch.device) -> tuple:
    player = random.choice([1, -1])
    gameState, _, _ = initializeGame(WIDTH, HEIGHT)
    while True:
        if player == 1:
            move = MCTSfindMove(gameState, player, SIMULATIONS,
                                UCB1, model=None, device=None, cutoff=False)

        elif player == -1:
            move = MCTSfindMove(gameState, player, SIMULATIONS,
                                UCB1, model=model, device=device, cutoff=True)

        row = makeMove(gameState, player, move)
        player = nextPlayer(player)
        resolveEvent(gameState, 0, WIDTH)

        if gameEnd(gameState, row, move).any():
            return 0 if player == -1 else 2
        elif not availableMoves(gameState):
            return 1


def main() -> None:
    result = [0, 0, 0]
    with torch.no_grad():
        model, device = loadModel(file=FILE)

        for i in range(100):
            res = validationGame(SIMULATIONS, model, device)
            result[res] += 1
            print(result)
        print('Wins player 1, Draws, Wins player -1')

        # row, col = game(gameState, screen, frame, SIMULATIONS, model, device)
        # if not gameOver(screen, gameEnd(gameState, row, col), WIDTH):
        #     main()


if __name__ == '__main__':
    main()
