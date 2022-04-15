from interface import initializeGame, resolveEvents, draw, gameOver
from gameplay import nextPlayer, makeMove
from MCTS import AIfindMove

# variables
GRIDSIZE = 9  # should be odd
MATRIX_SIZE = 2*GRIDSIZE+1
WIDTH = 100

SIMULATIONS = 3
CUTOFF = 0


def main() -> None:
    players, gameBoard, wall, screen, font = initializeGame(
        MATRIX_SIZE, GRIDSIZE, WIDTH)
    players[1].species = 'AI'
    draw(screen, players, wall, MATRIX_SIZE, WIDTH)
    while True:
        if players[0].species == 'human':
            if resolveEvents(screen, players, wall, gameBoard, WIDTH):
                nextPlayer(players, wall)

        else:  # players[0].species == 'AI'
            move = AIfindMove(gameBoard, players,
                              SIMULATIONS, CUTOFF, MATRIX_SIZE)
            player = players[0]
            makeMove(gameBoard, move, player.num)
            if move[0] == 'walk':
                player.r, player.c = move[1][1]
            else:  # move[0] == 'wall'
                for row, col in move[1]:
                    player.walls.append((row, col))

            nextPlayer(players, wall)

        draw(screen, players, wall, MATRIX_SIZE, WIDTH)

        if players[-1].winner():
            gameOver(players, screen, font, GRIDSIZE, WIDTH)


if __name__ == "__main__":
    main()
