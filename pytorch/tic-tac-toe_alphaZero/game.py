import pygame
import torch
from AI import MCTS_find_move, load_conv_model
from gameplay import (available_moves, game_end, game_result, make_move,
                      next_player)
from interface import (chooseConfig, draw, gameOver, initializeGame,
                       resolveEvent)
from MCTSData import MCTSData

# Configurations
SIMULATIONS = 10
WIDTH = 200
UCB1 = 1.4


def game(data: MCTSData, screen: pygame.surface.Surface,
         frame: pygame.Surface) -> int:
    while True:
        if data.player == -1:
            # Human
            move = resolveEvent(data.board, data.player, WIDTH)
            make_move(data.board, data.player, move)
            if move:
                data.player = next_player(data.player)

        elif data.player == 1:
            # AI
            move = MCTS_find_move(data)
            # move = best_evaluation_find_move(data)
            make_move(data.board, data.player, move)
            data.player = next_player(data.player)
            resolveEvent(data.board, 0, WIDTH)

            print(torch.softmax(data.model(
                data.board, data.player, data.device), dim=2))

        draw(screen, frame, data.board, WIDTH, data.player)

        if game_end(data.board):
            return game_result(data.board)

        if not available_moves(data.board):
            return 0


def main() -> None:
    sims = chooseConfig(SIMULATIONS)
    player = -1
    board, screen, frame = initializeGame(WIDTH)
    draw(screen, frame, board, WIDTH)
    model, device = load_conv_model()

    data = MCTSData(board, player, UCB1, model, device, sim_number=sims)

    result = game(data, screen, frame)
    if not gameOver(screen, result, WIDTH):
        main()


if __name__ == '__main__':
    main()
