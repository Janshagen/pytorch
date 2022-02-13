from Player import Player
from Wall import Wall
from Line import Line
import pygame

pygame.init()

gridsize = 9  # Bör vara udda
middle = int((gridsize - 1) / 2)
w = 80

# Asks how many people are playing until 2, 3 or 4 is given
players_found = False
while not players_found:
    try:
        number_of_players = int(input('How many people are playing? '))
        if number_of_players > 4 or number_of_players < 2:
            raise ValueError
        players_found = True
    except ValueError:
        print(' ')
        print('Please enter a number between 2 and 4')

# Initializes the players
if number_of_players == 2:
    players = [Player((0, 0, 255), gridsize - 1, middle, 'r0'),
               Player((255, 0, 0), 0, middle, 'r1')]
elif number_of_players == 3:
    players = [Player((0, 0, 255), gridsize - 1, middle, 'r0'),
               Player((255, 0, 0), 0, middle, 'r1'),
               Player((0, 255, 0), middle, 0, 'c1')]
elif number_of_players == 4:
    players = [Player((0, 0, 255), gridsize - 1, middle, 'r0'),
               Player((255, 0, 0), 0, middle, 'r1'),
               Player((0, 255, 0), middle, 0, 'c1'),
               Player((255, 255, 0), middle, gridsize - 1, 'c0')]
players[0].current = True
wall = Wall(players[0].color)

# Initializing screen, font, clock and the grids containing the lines
screen = pygame.display.set_mode((gridsize * w + 1, gridsize * w + 1))
font = pygame.font.Font(None, 128)
clock = pygame.time.Clock()
horizontal_lines = [[Line(r, c, 'horizontal') for c in range(gridsize + 1)] for r in range(gridsize + 1)]
vertical_lines = [[Line(r, c, 'vertical') for c in range(gridsize + 1)] for r in range(gridsize + 1)]
tag_index = 1

# Sets the boundaries of the board to occupied
for row in horizontal_lines:
    for line in row:
        if line.r == 0 or line.r == gridsize:
            line.occ = True

for col in vertical_lines:
    for line in col:
        if line.c == 0 or line.c == gridsize:
            line.occ = True


def take_turn(player_, w, display):
    """Checks the event queue for mouse press or button press. Mouse press will place a wall if that space is available
    for a wall, i.e. not on top of another wall but right next to it. Space bar will rotate the wall.
    The arrow keys will move the current player. And pressing the exit button will close the window."""
    global running
    for event_ in pygame.event.get():
        if event_.type == pygame.MOUSEBUTTONDOWN:
            if place_wall(wall.orientation):
                next_player(player_)
                return
        if event_.type == pygame.KEYDOWN:
            if event_.key == pygame.K_SPACE:
                wall.flip()
            else:
                opponents = [(opponent.r, opponent.c) for opponent in players if not opponent.current]
                if player_.move(event_, horizontal_lines, vertical_lines, opponents, w, display):
                    next_player(player_)
                    return
        if event_.type == pygame.QUIT:
            running = False
            return


def place_wall(orientation):
    global tag_index
    if orientation == 'horizontal' and not horizontal_lines[wall.r][wall.c].occ and \
            not horizontal_lines[wall.r][wall.c - 1].occ and \
            not (vertical_lines[wall.r][wall.c].tag == vertical_lines[wall.r - 1][wall.c].tag and
                 vertical_lines[wall.r - 1][wall.c].occ):
        horizontal_lines[wall.r][wall.c].paint(wall.color, tag_index)
        horizontal_lines[wall.r][wall.c - 1].paint(wall.color, tag_index)
        tag_index += 1
        return True

    elif wall.orientation == 'vertical' and not vertical_lines[wall.r][wall.c].occ and \
            not vertical_lines[wall.r - 1][wall.c].occ and \
            not (horizontal_lines[wall.r][wall.c].tag == horizontal_lines[wall.r][wall.c - 1].tag and
                 horizontal_lines[wall.r][wall.c - 1].occ):
        vertical_lines[wall.r][wall.c].paint(wall.color, tag_index)
        vertical_lines[wall.r - 1][wall.c].paint(wall.color, tag_index)
        return True


def next_player(player_):
    """Changes the current attribute to the next player in players"""
    player_.current = False
    index = players.index(player_)
    if index == len(players) - 1:
        players[0].current = True
        wall.color = players[0].color
    else:
        players[index + 1].current = True
        wall.color = players[index + 1].color
    return

# An attempt to not allow players to be trapped
# def trapped():
#     for i in range(1, gridsize):
#         if horizontal_lines[i][0].occ:
#             if check_next(i, 0, horizontal_lines):
#                 print('jhgfkjabfkjabfakfbaykfbyuakfb')
#
#
# def check_next(r, c, matrix):
#     print(c)
#     if c == gridsize - 1:
#         pass
#     elif matrix[r][c + 1].occ:
#         check_next(r, c + 1, matrix)
#     else:
#         return False


winner = False
running = True
while running:
    if not winner:
        clock.tick(10)
        screen.fill((0, 0, 0))

        for row in horizontal_lines:
            for line in row:
                line.show(screen, w)

        for col in vertical_lines:
            for line in col:
                line.show(screen, w)

        wall.move(gridsize, w)
        wall.show(screen, w)

        for player in players:
            if player.current:
                take_turn(player, w, screen)
                if player.winner(gridsize):
                    winner = True
                    win = font.render('Winner', True, player.color)
            player.show(screen, w)
    else:
        if pygame.event.get(pygame.QUIT):
            running = False
        screen.blit(win, (205, 160))

    pygame.event.pump()
    pygame.display.flip()
