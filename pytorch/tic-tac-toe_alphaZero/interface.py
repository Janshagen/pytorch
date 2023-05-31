import sys

import numpy as np
import numpy.typing as npt

import pygame
from typing import Optional

from gameplay import availableMoves

WHITE = (230, 230, 230)
GREY = (180, 180, 180)
PURPLE = (138, 43, 226)
RED = (220, 20, 60)

board_type = npt.NDArray[np.int8]


def initializeGame(WIDTH: int
                   ) -> tuple[board_type,
                              pygame.surface.Surface, pygame.Surface]:
    gameState = np.zeros((3, 3), dtype=np.int8)

    pygame.init()
    screen = pygame.display.set_mode((3 * WIDTH, 3 * WIDTH))
    frame = pygame.Surface((3 * WIDTH, 3 * WIDTH))
    frame.fill(GREY)
    for i in range(3):
        for j in range(3):
            pygame.draw.rect(
                frame, WHITE, (i*WIDTH+3, j*WIDTH+3, WIDTH-6, WIDTH-6))
    frame.set_colorkey(WHITE)

    return gameState, screen, frame


def draw(screen: pygame.surface.Surface, frame: pygame.Surface,
         board: board_type, WIDTH: int, player: int = 0) -> None:

    screen.fill(WHITE)
    drawPieces(screen, board, player, WIDTH)
    screen.blit(frame, (0, 0))

    pygame.display.flip()


def drawPieces(screen: pygame.surface.Surface, board: board_type, player,
               WIDTH: int) -> None:
    # placed pieces
    for i, row in enumerate(board):
        for j, spot in enumerate(row):
            if spot == 1:
                pygame.draw.circle(screen, PURPLE,
                                   int2coord(j, i, WIDTH), WIDTH // 3)
            elif spot == -1:
                pygame.draw.circle(screen, RED,
                                   int2coord(j, i, WIDTH), WIDTH // 3)

    # color = PURPLE if player == 1 else RED
    # # moving piece
    # mPos = mousePos(WIDTH)
    # mPos = mPos if mPos != (None, None) else (1, 1)
    # pygame.draw.circle(screen, color,
    #                    int2coord(mPos[1], mPos[0], WIDTH), WIDTH // 3)


# def animatePiece(screen: pygame.Surface, frame: pygame.Surface,
#                   board: np.ndarray, col: int, player: int, w: int, h: int):
#     for row in range(6):
#         if board[row][col] != 0:
#             break

#     color = RED if player == 1 else PURPLE
#     y = 0
#     while y < (row+1)*h:
#         screen.fill(WHITE)
#         drawPieces(screen, board, player, w, h, (row, col))
#         pygame.draw.circle(screen, color,
#                            (w*col + w/2, y+h/2), w // 3)
#         screen.blit(frame, (0, h))
#         y += h*0.012
#         pygame.display.flip()


def resolveEvent(board: board_type, player: int,
                 WIDTH: int) -> Optional[tuple[int, int]]:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN and player:
            return placePiece(board, WIDTH)
    return None


def placePiece(board: np.ndarray, WIDTH: int) -> Optional[tuple[int, int]]:
    moves = availableMoves(board)
    mPos = mousePos(WIDTH)

    if mPos in moves:
        return mPos
    return None


def gameOver(screen: pygame.surface.Surface, result: int, WIDTH: int) -> bool:
    color = PURPLE if result == 1 else RED
    text = 'Draw' if result == 0 else 'Winner'
    font = pygame.font.Font(None, 128)
    while True:
        textFont = font.render(text, True, color)
        screen.blit(textFont, (1.5 * WIDTH - font.size(text)[0]/2, WIDTH*0.75))
        pygame.display.flip()
        for event_ in pygame.event.get():
            if event_.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event_.type == pygame.KEYDOWN and event_.key == pygame.K_SPACE:
                return False


def int2coord(i: int, j: int, w: int) -> tuple[int, int]:
    return w*i + w//2, w*j + w//2


def mousePos(WIDTH: int) -> tuple[Optional[int], Optional[int]]:
    mPos = pygame.mouse.get_pos()
    for i in range(7):
        for j in range(7):
            if WIDTH * i < mPos[1] < WIDTH * (i+1) and \
                    WIDTH * j < mPos[0] < WIDTH * (j+1):
                return (i, j)
    return None, None


def chooseConfig(SIMULATIONS: int) -> int:
    if len(sys.argv) == 1:
        return SIMULATIONS

    if len(sys.argv) == 2:
        try:
            sims = int(sys.argv[1])
        except ValueError:
            print(
                '\n Usage: \n No arguments; 1000 simulations \n One argument; \
                {Number of simulations (int)}')
            sys.exit()
        return sims

    print(
        '\n Usage: \n No arguments; 1000 simulations \n One argument; \
            {Number of simulations (int)}')
    sys.exit()
