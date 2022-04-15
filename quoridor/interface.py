from Player import Player
from Wall import Wall
from gameplay import place_wall
import numpy as np
import pygame
import sys


def resolveEvents(display, players, wall, board, WIDTH) -> bool:
    """Checks the event queue for mouse press or button press. Mouse press will place a wall if that space is available
    for a wall, i.e. not on top of another wall. Space bar will rotate the wall.
    The arrow keys will move the current player. And pressing the exit button will close the window."""
    for event_ in pygame.event.get():
        if event_.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event_.type == pygame.MOUSEBUTTONDOWN:
            return place_wall(players, wall, board)

        elif event_.type == pygame.KEYDOWN:
            key = event_.key
            if key == pygame.K_SPACE:
                wall.flip()

            elif key in (pygame.K_UP, pygame.K_DOWN):
                return players[0].move(
                    key, board, players, WIDTH, display
                )

            elif key in (pygame.K_RIGHT, pygame.K_LEFT):
                return players[0].move(
                    key, board, players, WIDTH, display
                )
    pygame.event.pump()


def draw(screen, players, wall, MATRIX_SIZE, WIDTH) -> None:
    drawBackground(screen, WIDTH)
    for player in players:
        player.show(screen, WIDTH)
    wall.move(MATRIX_SIZE, WIDTH)
    wall.show(screen, WIDTH)

    pygame.display.flip()


def drawBackground(screen, WIDTH) -> None:
    screen.fill((190, 190, 190))
    for i in range(10):
        # draw lines
        pygame.draw.line(screen, (230, 230, 230),
                         (0, i*WIDTH), (WIDTH*9, i*WIDTH), width=3)
        pygame.draw.line(screen, (230, 230, 230),
                         (i*WIDTH, 0), (i*WIDTH, WIDTH*9), width=3)


def findPlayers():
    # Asks how many people are playing until 2, 3 or 4 is given
    while True:
        try:
            number_of_players = int(input("How many people are playing? "))
            if number_of_players > 4 or number_of_players < 2:
                raise ValueError
            return number_of_players
        except ValueError:
            print(" ")
            print("Please enter a number between 2 and 4")


def initializePlayers(MATRIX_SIZE):
    number_of_players = 2  # findPlayers()

    MIDDLE = int((MATRIX_SIZE - 1) / 2)
    # Initializes the players
    if number_of_players == 2:
        players = [
            Player(1, (138, 43, 226), MATRIX_SIZE - 2, MIDDLE,
                   [(1, col) for col in range(MATRIX_SIZE)]),
            Player(2, (220, 20, 60), 1, MIDDLE, [
                   (MATRIX_SIZE-2, col) for col in range(MATRIX_SIZE)]),
        ]
    elif number_of_players == 3:
        players = [
            Player((138, 43, 226), MATRIX_SIZE - 2, MIDDLE,
                   [(1, col) for col in range(MATRIX_SIZE)]),
            Player((220, 20, 60), 1, MIDDLE, [
                   (MATRIX_SIZE-2, col) for col in range(MATRIX_SIZE)]),
            Player((235, 200, 0), MIDDLE, 1, [
                   (row, MATRIX_SIZE-2) for row in range(MATRIX_SIZE)]),
        ]
    elif number_of_players == 4:
        players = [
            Player((138, 43, 226), MATRIX_SIZE - 2, MIDDLE,
                   [(1, col) for col in range(MATRIX_SIZE)]),
            Player((220, 20, 60), 1, MIDDLE, [
                   (MATRIX_SIZE-2, col) for col in range(MATRIX_SIZE)]),
            Player((235, 200, 0), MIDDLE, 1, [
                   (row, MATRIX_SIZE-2) for row in range(MATRIX_SIZE)]),
            Player((50, 205, 50), MIDDLE, MATRIX_SIZE - 2,
                   [(row, 1) for row in range(MATRIX_SIZE)]),
        ]

    return players


def initializeGameBoard(players, MATRIX_SIZE):
    # 0-tomt, 1-4-spelare, 5-vägg
    board = np.zeros((MATRIX_SIZE, MATRIX_SIZE))
    for i in range(MATRIX_SIZE):
        board[i][0] = 5
        board[i][-1] = 5
        board[0][i] = 5
        board[-1][i] = 5

    for player in players:
        board[player.r][player.c] = player.num
    return board


def initializeGame(MATRIX_SIZE, GRIDSIZE, WIDTH):
    players = initializePlayers(MATRIX_SIZE)
    wall = Wall(players[0].color)
    gameBoard = initializeGameBoard(players, MATRIX_SIZE)

    # Initializing screen, font, clock and board
    pygame.init()
    screen = pygame.display.set_mode(
        (GRIDSIZE * WIDTH + 1, GRIDSIZE * WIDTH + 1))
    font = pygame.font.Font(None, 128)
    return players, gameBoard, wall, screen, font


def gameOver(players, screen, font, GRIDSIZE, WIDTH):
    while True:
        win = font.render("Winner", True, players[-1].color)
        screen.blit(win, (GRIDSIZE * WIDTH * 0.333, GRIDSIZE * WIDTH * 0.24))
        pygame.display.flip()
        for event_ in pygame.event.get():
            if event_.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
