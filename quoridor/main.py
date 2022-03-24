from Player import Player
from Wall import Wall
from Line import Line
import pygame, sys


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


def initializePlayers():
    number_of_players = findPlayers()
    # Initializes the players
    if number_of_players == 2:
        players = [
            Player((0, 0, 255), GRIDSIZE - 1, MIDDLE, "r0"),
            Player((255, 0, 0), 0, MIDDLE, "r1"),
        ]
    elif number_of_players == 3:
        players = [
            Player((0, 0, 255), GRIDSIZE - 1, MIDDLE, "r0"),
            Player((255, 0, 0), 0, MIDDLE, "r1"),
            Player((0, 255, 0), MIDDLE, 0, "c1"),
        ]
    elif number_of_players == 4:
        players = [
            Player((0, 0, 255), GRIDSIZE - 1, MIDDLE, "r0"),
            Player((255, 0, 0), 0, MIDDLE, "r1"),
            Player((0, 255, 0), MIDDLE, 0, "c1"),
            Player((255, 255, 0), MIDDLE, GRIDSIZE - 1, "c0"),
        ]

    return players


def initializeGameBoard():
    horizontal_lines = [
        [Line(r, c, "horizontal") for c in range(GRIDSIZE + 1)]
        for r in range(GRIDSIZE + 1)
    ]
    vertical_lines = [
        [Line(r, c, "vertical") for c in range(GRIDSIZE + 1)]
        for r in range(GRIDSIZE + 1)
    ]

    # Sets the boundaries of the board to occupied
    for row in horizontal_lines:
        for line in row:
            if line.r == 0 or line.r == GRIDSIZE:
                line.occ = True

    for col in vertical_lines:
        for line in col:
            if line.c == 0 or line.c == GRIDSIZE:
                line.occ = True

    return horizontal_lines, vertical_lines


def initializeGame():
    players = initializePlayers()
    wall = Wall(players[0].color)
    tag_index = 1
    horizontal_lines, vertical_lines = initializeGameBoard()

    # Initializing screen, font, clock and the grids containing the lines
    pygame.init()
    screen = pygame.display.set_mode((GRIDSIZE * WIDTH + 1, GRIDSIZE * WIDTH + 1))
    font = pygame.font.Font(None, 128)
    clock = pygame.time.Clock()
    return (
        players,
        horizontal_lines,
        vertical_lines,
        wall,
        tag_index,
        screen,
        font,
        clock,
    )


def resolveEvents(
    display, players, wall, horizontal_lines, vertical_lines, tag_index
) -> bool:
    """Checks the event queue for mouse press or button press. Mouse press will place a wall if that space is available
    for a wall, i.e. not on top of another wall. Space bar will rotate the wall.
    The arrow keys will move the current player. And pressing the exit button will close the window."""
    for event_ in pygame.event.get():
        if event_.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event_.type == pygame.MOUSEBUTTONDOWN:
            return place_wall(wall, horizontal_lines, vertical_lines, tag_index)

        elif event_.type == pygame.KEYDOWN:
            key = event_.key
            if key == pygame.K_SPACE:
                wall.flip()

            elif key in (pygame.K_UP, pygame.K_DOWN):
                return players[0].move(
                    key, horizontal_lines, vertical_lines, players, WIDTH, display
                )

            elif key in (pygame.K_RIGHT, pygame.K_LEFT):
                return players[0].move(
                    key, vertical_lines, horizontal_lines, players, WIDTH, display
                )


def place_wall(wall, horizontal_lines, vertical_lines, tag_index) -> bool:
    if wall.orientation == "horizontal":
        primaryLines, secondaryLines = horizontal_lines, vertical_lines
        dir = (0, -1)
    else:  # wall.orientation == "vertical":
        primaryLines, secondaryLines = vertical_lines, horizontal_lines
        dir = (-1, 0)

    if (
        not primaryLines[wall.r][wall.c].occ
        and not primaryLines[wall.r + dir[0]][wall.c + dir[1]].occ
        and not (
            secondaryLines[wall.r][wall.c].tag
            == secondaryLines[wall.r + dir[1]][wall.c + dir[0]].tag
            and secondaryLines[wall.r + dir[1]][wall.c + dir[0]].occ
        )
    ):
        primaryLines[wall.r][wall.c].place(wall.color, wall.tag)
        primaryLines[wall.r + dir[0]][wall.c + dir[1]].place(wall.color, wall.tag)
        wall.tag += 1
        return True


def next_player(players, wall) -> None:
    """Changes order of the players list"""
    currentPlayer = players.pop(0)
    players.append(currentPlayer)
    wall.color = players[0].color


def draw(screen, horizontal_lines, vertical_lines, players, wall) -> None:
    screen.fill((0, 0, 0))

    for row in horizontal_lines:
        for line in row:
            line.show(screen, WIDTH)

    for col in vertical_lines:
        for line in col:
            line.show(screen, WIDTH)

    for player in players:
        player.show(screen, WIDTH)

    wall.move(GRIDSIZE, WIDTH)
    wall.show(screen, WIDTH)
    pygame.display.flip()
    
    
def gameOver():
    while True:
        win = font.render("Winner", True, players[-1].color)
        screen.blit(win, (205, 160))
        pygame.display.flip()
        for event_ in pygame.event.get():
            if event_.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        

def main() -> None:
    while True:
        clock.tick(10)
        draw(screen, horizontal_lines, vertical_lines, players, wall)

        if resolveEvents(screen, players, wall, horizontal_lines, vertical_lines, tag_index):
            next_player(players, wall)
            
        if players[-1].winner(GRIDSIZE):
            gameOver()

        pygame.event.pump()


if __name__ == "__main__":
    # variables
    GRIDSIZE = 9  # should be odd
    MIDDLE = int((GRIDSIZE - 1) / 2)
    WIDTH = 80

    (
        players,
        horizontal_lines,
        vertical_lines,
        wall,
        tag_index,
        screen,
        font,
        clock,
    ) = initializeGame()

    main()
