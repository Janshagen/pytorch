from Block import Block
from Cell import Cell
import pygame
import random
pygame.init()

w = 30
rowN = 20
colN = 10

# Skapar spelplanen
screen = pygame.display.set_mode((colN * w + 200, rowN * w + 1))
font = pygame.font.Font(None, 72)
next_block = pygame.font.Font(None, 36)
grid = [[Cell(i, j, w) for j in range(colN)] for i in range(rowN + 1)]
for cell in grid[-1][:]:
    cell.occ = True


def draw_grid(display):
    global rowN, colN, w
    for n in range(rowN + 1):
        if n == rowN:
            color = (255, 0, 0)
        else:
            color = (255, 255, 255)
        pygame.draw.line(display, color, (0, n * w), (colN * w + 1, n * w))
    for n in range(colN + 1):
        if n == 0 or n == colN:
            color = (255, 0, 0)
        else:
            color = (255, 255, 255)
        pygame.draw.line(display, color, (n * w, 0), (n * w, rowN * w + 1))


def remove():
    """Tar bort en rad ifall den är full"""
    global colN, rowN, score
    for i, row_ in enumerate(grid[0:rowN]):
        occ = []
        for j, cell_ in enumerate(row_):
            if cell_.occ:
                occ.append(1)
        if occ.count(1) == colN:
            score += 1
            del grid[i]
            for n in range(i):
                n = i - n - 1
                for k in range(colN):
                    grid[n][k].i += 1
            grid.insert(0, [Cell(0, j, w)for j in range(colN)])
            remove()
            return


def restart():
    global block, score, time
    for row_ in grid[0:rowN]:
        for cell_ in row_:
            cell_.occ = False
            cell_.color = (0, 0, 0)
            block = spawn_block()
            score = 0
            time.clear()


def spawn_block(choose=None):
    """Skapar ett nytt block"""
    global w
    if choose:
        sort = choose
    else:
        sort = random.choice(sorts)
    block_ = Block(sort, [3, 0], w)
    return block_


def dead():
    """Kollar ifall spelaren har förlorat"""
    for n in range(colN):
        if grid[0][n].occ:
            return True


sorts = ("Z", "L", "S", "O", "I", "J", "T")
blocks = [spawn_block() for _ in range(4)]
blocks[0].active = True

clock = pygame.time.Clock()
score = 0
time = []
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    if not dead():
        screen.fill((0, 0, 0))
        clock.tick(30)
        time.append(1)
        blocks[0].outside(colN)
        blocks[0].check_down(grid)
        blocks[0].move(time, grid, screen, colN)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                cell.show(screen)
        for i, block in enumerate(blocks):
            block.show(screen, i)
        draw_grid(screen)
        screen.blit(next_block.render('Next Block', True, (255, 255, 255)), (340, 20))

        if not blocks[0].active:
            remove()
            blocks.pop(0)
            blocks.append(spawn_block())
            blocks[0].active = True
            time.clear()
        pygame.display.flip()
    else:
        screen.blit(font.render('Score: ' + str(score), True, (255, 255, 255)), (52, 138))
        pygame.display.flip()
        for event in pygame.event.get(pygame.KEYDOWN):
            if event.key == pygame.K_SPACE:
                restart()
