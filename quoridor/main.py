from interface import initializeGame, resolveEvents, draw, gameOver
from gameplay import nextPlayer


def main() -> None:
    players, gameBoard, wall, screen, font, clock = initializeGame(
        MATRIX_SIZE, GRIDSIZE, WIDTH)
    while True:
        if resolveEvents(screen, players, wall, gameBoard, WIDTH):
            nextPlayer(players, wall)

        clock.tick(10)
        draw(screen, players, wall, MATRIX_SIZE, WIDTH)

        if players[-1].winner():
            gameOver(players, screen, font, GRIDSIZE, WIDTH)


if __name__ == "__main__":
    # variables
    GRIDSIZE = 9  # should be odd
    MATRIX_SIZE = 2*GRIDSIZE+1
    WIDTH = 100
    main()
