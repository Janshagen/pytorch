from gameplay import availableMoves, gameEnd, makeMove, nextPlayer
from interface import draw, gameOver, initializeGame, resolveEvent, chooseConfig
from MCTS import AIfindMove
import torch
from train_model import Model

# Configurations
SIMULATIONS = 1000
WIDTH = 200
UCB1 = 1.4

FILE = 'tic-tac-toe-model.pth'

INPUT_SIZE = 9
OUTPUT_SIZE = 1
HIDDEN_SIZE1 = 54
HIDDEN_SIZE2 = 54


def game(gameState, player, screen, frame, sims) -> int:
    while True:
        if player == 1:
            # Human
            move = resolveEvent(gameState, player, WIDTH)
            makeMove(gameState, player, move)
            if move:
                player = nextPlayer(player)

        elif player == -1:
            # AI
            move = AIfindMove(gameState, player, sims, UCB1)
            makeMove(gameState, player, move)
            player = nextPlayer(player)
            resolveEvent(gameState, 0, WIDTH)

        draw(screen, frame, gameState, WIDTH, move, player)

        if not availableMoves(gameState):
            return 0

        if gameEnd(gameState).any():
            return nextPlayer(player)


def main() -> None:
    sims = chooseConfig(SIMULATIONS)
    player = 1
    gameState, screen, frame = initializeGame(WIDTH)
    draw(screen, frame, gameState, WIDTH)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = Model(INPUT_SIZE, HIDDEN_SIZE1,
    #               HIDDEN_SIZE2, OUTPUT_SIZE).to(device)
    # model.load_state_dict(torch.load(FILE))
    # model.to(device)
    # model.eval()

    result = game(gameState, player, screen, frame, sims)
    if not gameOver(screen, result, WIDTH):
        main()


if __name__ == '__main__':
    main()
