import random

import numpy as np
import pygame
import torch

from AI import MCTSfindMove, loadModel
from gameplay import gameEnd, makeMove, nextPlayer
from interface import (chooseConfig, draw, gameOver, initializeGame,
                       resolveEvent)

# Configurations
SIMULATIONS = 1000
WIDTH = 120
HEIGHT = WIDTH*0.8
UCB1 = 1.4


def game(gameState: np.ndarray, player: int, screen: pygame.Surface, frame: pygame.Surface, sims: int, model: torch.nn.Module, device: torch.device) -> tuple:
    while True:
        if player == 1:
            # Human
            move = resolveEvent(gameState, player, WIDTH)
            row = makeMove(gameState, player, move)
            if type(move) == int:
                player = nextPlayer(player)

        elif player == -1:
            # AI
            print('Prediction before AIs move:', torch.softmax(model(model.board2tensor(
                gameState, player, device))[0][0], dim=0))
            move = MCTSfindMove(gameState, player, sims, UCB1, model, device)
            row = makeMove(gameState, player, move)
            player = nextPlayer(player)
            resolveEvent(gameState, 0, WIDTH)
            print('Prediction after  AIs move:', torch.softmax(model(model.board2tensor(
                gameState, player, device))[0][0], dim=0))

        draw(screen, frame, gameState, WIDTH, HEIGHT, move, player)
        if gameEnd(gameState, row, move).any():
            return (row, move)


def main() -> None:
    sims = chooseConfig(SIMULATIONS)
    player = random.choice([1, -1])
    gameState, screen, frame = initializeGame(WIDTH, HEIGHT)
    draw(screen, frame, gameState, WIDTH, HEIGHT)
    with torch.no_grad():
        model, device = loadModel()
        row, col = game(gameState, player, screen, frame, sims, model, device)
        if not gameOver(screen, gameEnd(gameState, row, col), WIDTH):
            main()


if __name__ == '__main__':
    main()
