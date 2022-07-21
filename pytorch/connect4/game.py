import random

import numpy as np
import pygame
import torch

from AI import MCTSfindMove, loadModel
from gameplay import availableMoves, gameEnd, makeMove, nextPlayer
from interface import (chooseConfig, draw, gameOver, initializeGame,
                       resolveEvent)

# Configurations
SIMULATIONS = 10000
WIDTH = 120
HEIGHT = WIDTH*0.8
UCB1 = 1.4


def game(gameState: np.ndarray, player: int, screen: pygame.Surface, frame: pygame.Surface, sims: int, model: torch.nn.Module, device: torch.device) -> tuple:
    while True:
        if player == 1:
            # MCTS
            # move = MCTSfindMove(gameState, player, sims,
            #                     UCB1, model, device, cutoff=False)
            # row = makeMove(gameState, player, move)
            # player = nextPlayer(player)
            # resolveEvent(gameState, 0, WIDTH)

            # Human
            move = resolveEvent(gameState, player, WIDTH)
            row = makeMove(gameState, player, move)
            if type(move) == int:
                player = nextPlayer(player)

        elif player == -1:
            # Neural network
            print('Prediction after human move:', torch.softmax(model(model.board2tensor(
                gameState, player, device))[0][0], dim=0))
            move = MCTSfindMove(gameState, player, sims,
                                UCB1, model, device, cutoff=True)
            row = makeMove(gameState, player, move)
            player = nextPlayer(player)
            resolveEvent(gameState, 0, WIDTH)
            print(f'Prediction after AIs move:', torch.softmax(model(model.board2tensor(
                gameState, player, device))[0][0], dim=0))

        draw(screen, frame, gameState, WIDTH, HEIGHT, move, player)
        if gameEnd(gameState, row, move).any():
            # return 0 if player == -1 else 2

            return (row, move)

        if not availableMoves(gameState):
            # return 1

            return (row, move)


def main() -> None:
    result = [0, 0, 0]
    sims = chooseConfig(SIMULATIONS)
    player = random.choice([1, -1])
    gameState, screen, frame = initializeGame(WIDTH, HEIGHT)
    draw(screen, frame, gameState, WIDTH, HEIGHT)
    with torch.no_grad():
        model, device = loadModel()

        # for i in range(100):
        #     gameState, screen, frame = initializeGame(WIDTH, HEIGHT)
        #     res = game(gameState, player, screen, frame, sims, model, device)
        #     result[res] += 1
        #     print(result)

        row, col = game(gameState, player, screen, frame, sims, model, device)
        if not gameOver(screen, gameEnd(gameState, row, col), WIDTH):
            main()


if __name__ == '__main__':
    main()
