import pygame
import torch
from AI import MCTSfindMove, loadConvModel
from gameplay import availableMoves, gameEnd, makeMove, nextPlayer
from interface import (chooseConfig, draw, gameOver, initializeGame,
                       resolveEvent)
from MCTSData import MCTSData

# Configurations
SIMULATIONS = 100
WIDTH = 200
UCB1 = 1.4


def game(data: MCTSData, screen: pygame.surface.Surface,
         frame: pygame.Surface) -> int:
    while True:
        if data.player == -1:
            # Human
            move = resolveEvent(data.board, data.player, WIDTH)
            makeMove(data.board, data.player, move)
            if move:
                data.player = nextPlayer(data.player)

        elif data.player == 1:
            # AI
            move = MCTSfindMove(data)
            # move = bestEvaluationFindMove(data, model, device)
            makeMove(data.board, data.player, move)
            data.player = nextPlayer(data.player)
            resolveEvent(data.board, 0, WIDTH)

            print(torch.softmax(data.model(
                data.board, data.player, data.device), dim=2))

        draw(screen, frame, data.board, WIDTH, data.player)

        if gameEnd(data.board):
            return nextPlayer(data.player)

        if not availableMoves(data.board):
            return 0


def main() -> None:
    sims = chooseConfig(SIMULATIONS)
    player = -1
    board, screen, frame = initializeGame(WIDTH)
    draw(screen, frame, board, WIDTH)
    model, device = loadConvModel()

    data = MCTSData(board, player, UCB1, model, device, sim_number=sims)

    result = game(data, screen, frame)
    if not gameOver(screen, result, WIDTH):
        main()


if __name__ == '__main__':
    main()
