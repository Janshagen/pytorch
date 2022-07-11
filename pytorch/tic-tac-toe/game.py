from gameplay import availableMoves, gameEnd, makeMove, nextPlayer
from interface import (chooseConfig, draw, gameOver, initializeGame,
                       resolveEvent)
from AI import MCTSfindMove, bestEvaluationFindMove, loadLinearModel, loadConvModel
import torch

# Configurations
SIMULATIONS = 100
WIDTH = 200
UCB1 = 1.4


def game(gameState, model, device, player, screen, frame, sims) -> int:
    while True:
        if player == -1:
            # Human
            move = resolveEvent(gameState, player, WIDTH)
            makeMove(gameState, player, move)
            if move:
                player = nextPlayer(player)

        elif player == 1:
            # AI
            # move = MCTSfindMove(gameState, player, sims, UCB1, model, device)
            move = bestEvaluationFindMove(gameState, player, model, device)
            makeMove(gameState, player, move)
            player = nextPlayer(player)
            resolveEvent(gameState, 0, WIDTH)
            print(torch.softmax(model(model.board2tensor(
                gameState, nextPlayer(player), device)), dim=2))

        draw(screen, frame, gameState, WIDTH, move, player)

        if gameEnd(gameState).any():
            return nextPlayer(player)

        if not availableMoves(gameState):
            return 0


def main() -> None:
    sims = chooseConfig(SIMULATIONS)
    player = -1
    gameState, screen, frame = initializeGame(WIDTH)
    draw(screen, frame, gameState, WIDTH)
    model, device = loadConvModel()

    result = game(gameState, model, device, player, screen, frame, sims)
    if not gameOver(screen, result, WIDTH):
        main()


if __name__ == '__main__':
    main()
