import pygame
import sys
from typing import Optional

from GameRules import YatzyGameState, Sheet, Die


class InterfaceConnect4:

    WHITE = (230, 230, 230)
    GREY = (180, 180, 180)
    PURPLE = (138, 43, 226)
    RED = (220, 20, 60)

    def __init__(self, WIDTH: int):
        self.WIDTH = WIDTH
        self.circle_radius = WIDTH//3
        self.num_rows = 17

        self.dice_positions = [(7*WIDTH-8, (3*i+2)*WIDTH) for i in range(5)]

        pygame.init()
        self.screen = pygame.display.set_mode((8 * WIDTH, self.num_rows * WIDTH))
        self.frame = self.create_frame()

        self.score_font = pygame.font.Font(None, 32)
        self.die_font = pygame.font.Font(None, 64)

    def create_frame(self):
        frame = pygame.Surface((6 * self.WIDTH + 3, self.num_rows * self.WIDTH))
        frame.fill(self.GREY)
        # pygame.draw.rect(frame, self.WHITE, (1*self.WIDTH-3, 0,
        #                                      6, self.num_rows*self.WIDTH))
        scale = 1
        for i in range(6):
            if i == 3:
                scale = 0.5
            for j in range(self.num_rows):
                pygame.draw.rect(frame, self.WHITE,
                                 (i*self.WIDTH+3, j*self.WIDTH+3,
                                  int(scale*2*self.WIDTH)-6, self.WIDTH-6))
        frame.set_colorkey(self.WHITE)
        return frame

    def int2coord(self, i: int, j: int) -> tuple[float, float]:
        return (self.WIDTH*i + self.WIDTH/2, self.WIDTH*j + self.WIDTH/2)

    def draw(self, game_state: YatzyGameState) -> None:
        self.screen.fill(self.WHITE)
        self.write_numbers(game_state)
        self.write_move_names(game_state)
        self.screen.blit(self.frame, (0, 0))
        pygame.display.flip()

    def write_numbers(self, game_state: YatzyGameState) -> None:
        for i, die in enumerate(game_state.get_dice()):
            self.write_die_value(die, i)
        for sheet in game_state.sheets:
            self.write_scores(sheet)

    def write_die_value(self, die: Die, i: int):
        color = (0, 200, 0) if die.keeping else 255
        die_text = self.die_font.render(str(die.value), True, color)
        self.screen.blit(die_text, self.dice_positions[i])

    def write_scores(self, sheet: Sheet):
        for move in sheet.points:
            if move in sheet.moves_left:
                continue
            self.write_score(move, sheet)

    def write_score(self, move: str, sheet: Sheet):
        string_value = str(sheet[move])
        value_size = self.score_font.size(string_value)
        line_number = Sheet.move2index(move)

        score_text = self.score_font.render(string_value, True, 255)
        placement = ((sheet.player+4.5) * self.WIDTH-value_size[0]/2,
                     (line_number+1.5) * self.WIDTH-value_size[1]/2)

        self.screen.blit(score_text, placement)

    def write_move_names(self, game_state: YatzyGameState):
        sheet = game_state.sheets[0]
        for i, move in enumerate(sheet.points.keys()):
            self.write_move_name(move, i)

    def write_move_name(self, move: str, i: int):
        name_text = self.score_font.render(move, True, 255)
        name_size = self.score_font.size(move)
        placement = (0.5 * self.WIDTH, (i+1.5) * self.WIDTH-name_size[1]/2)
        self.screen.blit(name_text, placement)

    def resolve_event(self, game_state: YatzyGameState) -> Optional[str]:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                game_state.throw()
                return None

            if event.type == pygame.MOUSEBUTTONDOWN:
                return self.resolve_click_action(game_state)
        return None

    def resolve_click_action(self, game_state: YatzyGameState) -> Optional[str]:
        mouse_pos_x = pygame.mouse.get_pos()[0]
        if mouse_pos_x < 6*self.WIDTH:
            return self.choose_move(game_state)
        return self.save_die(game_state)

    def choose_move(self, game_state: YatzyGameState) -> Optional[str]:
        moves = list(game_state.available_moves())
        move_positions = [Sheet.move2index(move)+1 for move in moves]
        mouse_pos = self.mouse_index_position()

        for i, move_position in enumerate(move_positions):
            if mouse_pos == move_position:
                return moves[i]
        return None

    def save_die(self, game_state: YatzyGameState) -> None:
        for i, die in enumerate(game_state.get_dice()):
            if self.close_enough(i):
                die.flip_state()

    def close_enough(self, i: int):
        mouse_pos_y = pygame.mouse.get_pos()[1]
        return abs(mouse_pos_y - self.dice_positions[i][1]) < self.WIDTH

    def mouse_index_position(self) -> Optional[int]:
        mouse_pos_y = pygame.mouse.get_pos()[1]

        for i in range(self.num_rows):
            if self.WIDTH * i < mouse_pos_y <= self.WIDTH * (i+1):
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
