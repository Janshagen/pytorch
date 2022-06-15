from gameplay import gameEnd, makeMove, nextPlayer
from interface import draw, gameOver, initializeGame, resolveEvent, chooseConfig
from MCTS import AIfindMove

# Configurations
SIMULATIONS = 1000
WIDTH = 120
HEIGHT = WIDTH*0.8
UCB1 = 1.4


def game(gameState, player, screen, frame, sims):
    while True:
        if player == 1:
            # Human
            move = resolveEvent(gameState, player, WIDTH)
            makeMove(gameState, player, move)
            if type(move) == int:
                player = nextPlayer(player)

        elif player == 2:
            # AI
            move = AIfindMove(gameState, player, sims, UCB1)
            makeMove(gameState, player, move)
            player = nextPlayer(player)
            resolveEvent(gameState, 0, WIDTH)

        draw(screen, frame, gameState, WIDTH, HEIGHT, move, player)
        if gameEnd(gameState).any():
            return


def main() -> None:
    sims = chooseConfig(SIMULATIONS)
    player = 1
    gameState, screen, frame = initializeGame(WIDTH, HEIGHT)
    draw(screen, frame, gameState, WIDTH, HEIGHT)
    game(gameState, player, screen, frame, sims)
    if not gameOver(screen, gameEnd(gameState), WIDTH):
        main()


if __name__ == '__main__':
    main()
