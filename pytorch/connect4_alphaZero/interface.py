import pygame
import sys
from typing import Optional

from GameRules import Connect4GameState

WHITE = (230, 230, 230)
GREY = (180, 180, 180)
PURPLE = (138, 43, 226)
RED = (220, 20, 60)


def initialize_game(WIDTH: int, HEIGHT: int) -> \
        tuple[pygame.surface.Surface, pygame.Surface]:

    pygame.init()
    screen = pygame.display.set_mode((7 * WIDTH, 7 * HEIGHT))
    frame = pygame.Surface((7 * WIDTH, 6 * HEIGHT))
    frame.fill(GREY)
    for i in range(7):
        for j in range(6):
            pygame.draw.circle(frame, WHITE,
                               int2coord(i, j, WIDTH, HEIGHT), WIDTH//3)
    frame.set_colorkey(WHITE)

    return screen, frame


def int2coord(i: int, j: int, w: int, h: int) -> tuple[float, float]:
    return (w*i + w/2, h*j + h/2)


def draw(screen: pygame.surface.Surface, frame: pygame.Surface,
         game_state: Connect4GameState, WIDTH: int, HEIGHT: int,
         move: Optional[int] = None) -> None:
    if move:
        animate_piece(screen, frame, game_state, move, WIDTH, HEIGHT)

    screen.fill(WHITE)
    draw_pieces(screen, game_state, WIDTH, HEIGHT)
    screen.blit(frame, (0, HEIGHT))

    pygame.display.flip()


def animate_piece(screen: pygame.surface.Surface, frame: pygame.Surface,
                  game_state: Connect4GameState, move: int, w: int, h: int):
    row = 0
    for row in range(6):
        if game_state.board[row][move] != 0:
            break

    color = RED if game_state.player == 1 else PURPLE
    y = 0
    while y < (row+1)*h:
        screen.fill(WHITE)
        draw_pieces(screen, game_state, w, h, (row, move))
        pygame.draw.circle(screen, color,
                           (w*move + w/2, y+h/2), w // 3)
        screen.blit(frame, (0, h))
        y += h*0.012
        pygame.display.flip()


def draw_pieces(screen: pygame.surface.Surface, game_state: Connect4GameState,
                WIDTH: int, HEIGHT: int,
                lastPlaced: tuple = (None, None)) -> None:
    # placed pieces
    for i, row in enumerate(game_state.board):
        for j, spot in enumerate(row):
            if (i, j) == lastPlaced:
                continue
            elif spot == 1:
                pygame.draw.circle(screen, PURPLE,
                                   int2coord(j, i+1, WIDTH, HEIGHT), WIDTH // 3)
            elif spot == -1:
                pygame.draw.circle(screen, RED,
                                   int2coord(j, i+1, WIDTH, HEIGHT), WIDTH // 3)
    color = PURPLE if game_state.player == 1 else RED
    # moving piece
    mPos = mouse_position(WIDTH)
    mPos = mPos if mPos is not None else 3
    pygame.draw.circle(screen, color,
                       int2coord(mPos, 0, WIDTH, HEIGHT), WIDTH // 3)


def resolve_event(game_state: Connect4GameState, WIDTH: int) -> Optional[int]:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN:
            return place_piece(game_state, WIDTH)
    return None


def place_piece(game_state: Connect4GameState, WIDTH: int) -> Optional[int]:
    moves = game_state.available_moves()
    mouse_pos = mouse_position(WIDTH)

    if mouse_pos in moves:
        return mouse_pos
    return None


def mouse_position(WIDTH: int) -> Optional[int]:
    mPos = pygame.mouse.get_pos()[0]
    for i in range(7):
        if WIDTH * i < mPos <= WIDTH * (i+1):
            return i
    return None


def game_over(screen: pygame.surface.Surface, result: int, WIDTH: int) -> bool:
    color = PURPLE if result == 1 else RED
    font = pygame.font.Font(None, 128)
    while True:
        win = font.render("Winner", True, color)
        screen.blit(win, (3.5 * WIDTH - font.size('Winner')[0]/2, WIDTH*0.1))
        pygame.display.flip()
        for event_ in pygame.event.get():
            if event_.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event_.type == pygame.KEYDOWN and event_.key == pygame.K_SPACE:
                return False


def choose_config(SIMULATIONS: int) -> int:
    if len(sys.argv) == 1:
        return SIMULATIONS

    if len(sys.argv) == 2:
        try:
            sims = int(sys.argv[1])
        except ValueError:
            print(
                '\n Usage: \n No arguments; 1000 simulations \n \
                One argument; \
                {Number of simulations (int)}')
            sys.exit()
        return sims

    print(
        '\n Usage: \n No arguments; 1000 simulations \n One argument; \
        {Number of simulations (int)}')
    sys.exit()
