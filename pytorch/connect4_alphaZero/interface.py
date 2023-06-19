import pygame
import sys
from typing import Optional

from GameRules import Connect4GameState


class InterfaceConnect4:

    WHITE = (230, 230, 230)
    GREY = (180, 180, 180)
    PURPLE = (138, 43, 226)
    RED = (220, 20, 60)

    def __init__(self, WIDTH: int, HEIGHT: int):
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.circle_radius = WIDTH//3

        pygame.init()
        self.screen = pygame.display.set_mode((7 * WIDTH, 7 * HEIGHT))
        self.frame = self.create_frame()

    def create_frame(self):
        frame = pygame.Surface((7 * self.WIDTH, 6 * self.HEIGHT))
        frame.fill(self.GREY)
        for i in range(7):
            for j in range(6):
                pygame.draw.circle(frame, self.WHITE,
                                   self.int2coord(i, j), self.circle_radius)
        frame.set_colorkey(self.WHITE)
        return self.frame

    def draw_circle(self, color, i: int, j: int):
        pygame.draw.circle(self.screen, color, self.int2coord(i, j), self.circle_radius)

    def int2coord(self, i: int, j: int) -> tuple[float, float]:
        return (self.WIDTH*i + self.WIDTH/2, self.HEIGHT*j + self.HEIGHT/2)

    def draw(self, game_state: Connect4GameState, move: Optional[int] = None) -> None:
        if move:
            self.animate_piece(game_state, move)

        self.screen.fill(self.WHITE)
        self.draw_pieces(game_state)
        self.screen.blit(self.frame, (0, self.HEIGHT))
        pygame.display.flip()

    def animate_piece(self, game_state: Connect4GameState, move: int) -> None:
        row = 0
        for row in range(6):
            if game_state.board[row][move] != 0:
                break

        color = self.RED if game_state.player == 1 else self.PURPLE
        y = 0
        while y < (row+1)*self.HEIGHT:
            self.screen.fill(self.WHITE)
            self.draw_pieces(game_state, (row, move))
            pygame.draw.circle(self.screen, color, (self.WIDTH*move + self.WIDTH/2,
                               y+self.HEIGHT/2), self.circle_radius)
            self.screen.blit(self.frame, (0, self.HEIGHT))
            y += self.HEIGHT*0.012
            pygame.display.flip()

    def draw_pieces(self, game_state: Connect4GameState,
                    animated_piece: tuple = (None, None)) -> None:
        self.draw_placed_pieces(game_state, animated_piece)
        self.draw_moving_piece(game_state)

    def draw_placed_pieces(self, game_state: Connect4GameState,
                           animated_piece: tuple = (None, None)) -> None:
        for i, row in enumerate(game_state.board):
            for j, spot in enumerate(row):
                if (i, j) == animated_piece:
                    continue
                elif spot == 1:
                    self.draw_circle(self.PURPLE, j, i+1)
                elif spot == -1:
                    self.draw_circle(self.RED, j, i+1)

    def draw_moving_piece(self, game_state: Connect4GameState) -> None:
        color = self.PURPLE if game_state.player == 1 else self.RED
        mouse_pos = self.mouse_position()
        mouse_pos = mouse_pos if mouse_pos is not None else 3
        pygame.draw.circle(self.screen, color,
                           self.int2coord(mouse_pos, 0), self.circle_radius)

    def resolve_event(self, game_state: Connect4GameState) -> Optional[int]:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                return self.place_piece(game_state)
        return None

    def place_piece(self, game_state: Connect4GameState) -> Optional[int]:
        moves = game_state.available_moves()
        mouse_pos = self.mouse_position()

        if mouse_pos in moves:
            return mouse_pos
        return None

    def mouse_position(self) -> Optional[int]:
        mPos = pygame.mouse.get_pos()[0]
        for i in range(7):
            if self.WIDTH * i < mPos <= self.WIDTH * (i+1):
                return i
        return None

    def play_again(self, result: int) -> bool:
        color = self.PURPLE if result == 1 else self.RED
        font = pygame.font.Font(None, 128)
        while True:
            win = font.render("Winner", True, color)
            self.screen.blit(win, (3.5 * self.WIDTH - font.size('Winner')[0]/2,
                                   self.WIDTH*0.1))
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    return True

    def choose_config(self, SIMULATIONS: int) -> int:
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
