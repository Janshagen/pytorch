from Cell import Cell
import random
import pygame

gridsize = 25 #20
mine_percentage = 15 #15
w = 30

# Initierar ruta och skapar grid med celler
pygame.init()
screen = pygame.display.set_mode((gridsize * w, gridsize * w))
font = pygame.font.Font(None, 32)


def restart():
    cells_ = [[Cell(i, j, random.random() < mine_percentage/100) for j in range(gridsize + 2)]
              for i in range(gridsize + 2)]

# Tar bort minor vid kanterna
    for i in (0, gridsize + 1):
        for j in range(gridsize + 2):
            cells_[i][j].mine = False
            cells_[j][i].mine = False

# Räknar angränsande minor
    for i in range(1, gridsize + 1):
        for j in range(1, gridsize + 1):
            cells_[i][j].neighbor_count = Cell.neighbor(cells_[i][j], cells_)
    return cells_


# Ritar olika färger och siffror beroende på cellens tillstånd
def redraw():
    global screen
    for i in range(1, gridsize + 1):
        for j in range(1, gridsize + 1):
            if cells[i][j].picked:
                pygame.draw.rect(screen, (0, 0, 255), ((i - 1) * w + 1, (j - 1) * w + 1, w - 1, w - 1))
            else:
                if not cells[i][j].revealed:
                    pygame.draw.rect(screen, (255, 255, 255), ((i - 1) * w + 1, (j - 1) * w + 1, w - 1, w - 1))
                elif cells[i][j].revealed and cells[i][j].mine:
                    pygame.draw.rect(screen, (255, 0, 0), ((i - 1) * w + 1, (j - 1) * w + 1, w - 1, w - 1))
                elif cells[i][j].revealed and not cells[i][j].mine:
                    pygame.draw.rect(screen, (0, 255, 0), ((i - 1) * w + 1, (j - 1) * w + 1, w - 1, w - 1))
                    if cells[i][j].neighbor_count:
                        neighbor_mines = font.render(str(cells[i][j].neighbor_count), True, (0, 0, 0))
                        screen.blit(neighbor_mines, (int((i - 1) * w + w / 4), int((j - 1) * w + w / 4)))


# Hanterar musklick
def pressed():
    global game
    x, y = pygame.mouse.get_pos()
    x = x // w
    y = y // w
    if pygame.mouse.get_pressed()[0]:
        cells[x + 1][y + 1].picked = False
        cells[x + 1][y + 1].revealed = True
        if cells[x + 1][y + 1].mine:
            for i in range(1, gridsize + 1):
                for j in range(1, gridsize + 1):
                    cells[i][j].revealed = True
            redraw()
            game_over = pygame.font.Font(None, 132)
            over = game_over.render("Game Over", True, (0, 0, 0))
            screen.blit(over, (50, 165))
            game = False
        else:
            if not nonmines_left():
                for i in range(1, gridsize + 1):
                    for j in range(1, gridsize + 1):
                        cells[i][j].revealed = True
                redraw()
                game_over = pygame.font.Font(None, 132)
                over = game_over.render("Game Won", True, (0, 0, 0))
                screen.blit(over, (60, 165))
                game = False
            elif not cells[x + 1][y + 1].neighbor_count:
                check_neighbor(x + 1, y + 1)
    elif pygame.mouse.get_pressed()[2]:
        cells[x + 1][y + 1].picked = True


# Revealar grannar som inte är minor
def check_neighbor(i, j):
    for n in range(-1, 2):
        for k in range(-1, 2):
            if not cells[i + n][j + k].mine and not cells[i + n][j + k].revealed and \
                    j + k != 0 and j + k != gridsize + 1 and i + n != 0 and i + n != gridsize + 1:
                cells[i + n][j + k].revealed = True
                if not cells[i + n][j + k].neighbor_count:
                    check_neighbor(i + n, j + k)


# Räknar hur många icke-minor som är kvar
def nonmines_left():
    nonmines = 0
    for i in range(1, gridsize + 1):
        for j in range(1, gridsize + 1):
            if not cells[i][j].mine and not cells[i][j].revealed:
                nonmines += 1
    return nonmines


# Main loop
cells = restart()
running = True
game = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    if game:
        redraw()
        pressed()
        pygame.display.flip()
        pygame.time.delay(10)
    else:
        for event in pygame.event.get(pygame.KEYDOWN):
            if event.key == pygame.K_SPACE:
                cells = restart()
                game = True
