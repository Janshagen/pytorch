from Sorts import sorts
import pygame


def occupied(block, grid, direction):
    for i, row in enumerate(block.shape):
        for j, val in enumerate(row):
            if direction == 'L':
                a = - 1
            elif direction == 'R':
                a = 1
            if val and grid[block.pos[1] + i][block.pos[0] + j + a].occ:
                return True


class Block:
    def __init__(self, sort, pos, w):
        self.sort = sort
        self.color = sorts[sort]["color"]
        self.shape = sorts[sort]["shape"]
        self.annoying = sorts[sort]["annoying"]
        self.pos = pos
        self.w = w
        self.active = False

    def show(self, screen, y):
        """Draws the block"""
        for i, row in enumerate(self.shape):
            for j, val in enumerate(row):
                if val:
                    if self.active:
                        pygame.draw.rect(screen, self.color, ((self.pos[0] + j) * self.w,
                                                              (self.pos[1] + i - self.annoying) * self.w,
                                                              self.w, self.w))
                    else:
                        pygame.draw.rect(screen, self.color, ((self.pos[0] + j + 9) * self.w,
                                                              (self.pos[1] + i + 5 * y - 1) * self.w, self.w, self.w))

    def move(self, time, grid, screen, col):
        """Moves the block. Right or left if permitted by the surroundings and down after 2/3 of a second or
        when the down button is pressed. Also calls the turn function if up is pressed"""
        if self.sort == "I" and not self.annoying:
            a = 1
        else:
            a = 0
        for event in pygame.event.get(pygame.KEYDOWN):
            if event.key == pygame.K_RIGHT and self.pos[0] + len(self.shape[0]) - a < 10 \
                    and not occupied(self, grid, 'R'):
                self.pos[0] += 1
            elif event.key == pygame.K_LEFT and self.pos[0] + a > 0 and not occupied(self, grid, 'L'):
                self.pos[0] += -1
            elif event.key == pygame.K_DOWN:
                while self.active:
                    self.pos[1] += 1
                    self.show(screen, 0)
                    self.check_down(grid)
            elif event.key == pygame.K_UP and self.turn(col, grid=grid, test=True):
                self.turn(col)
        if time.count(1) > 20:
            self.pos[1] += 1
            time.clear()

    def turn(self, col, grid=(), test=False):
        """Rotates the block if permitted"""
        new = []
        for j, _ in enumerate(self.shape[0]):
            row = []
            for i, _ in enumerate(self.shape):
                i = len(self.shape) - i - 1
                row.append(self.shape[i][j])
            new.append(row)
        if test:
            return not self.test_turn(new, grid, col)
        else:
            self.shape = new
            if self.sort == "I" and self.annoying:
                self.pos[1] += 1
            if self.sort == "I":
                self.annoying = abs(self.annoying - 1)

    def test_turn(self, new, grid, col):
        for i, row in enumerate(new):
            for j, val in enumerate(row):
                if val and grid[self.pos[1] + i][self.pos[0] + j - self.offset(col)].occ:
                    return True

    def offset(self, col):
        if self.pos[0] + len(self.shape) > col:
            return self.pos[0] + len(self.shape) - col
        elif self.pos[0] < 0:
            return -1
        else:
            return 0

    def outside(self, col):
        """Checks if the block is outside the grid while turning"""
        if not self.sort == "I" or self.annoying:
            while self.pos[0] < 0:
                self.pos[0] += 1
            while self.pos[0] + len(self.shape[0]) > col:
                self.pos[0] -= 1

    def check_down(self, grid):
        """Checks if a block is directly underneath"""
        for i, row in enumerate(reversed(self.shape)):
            i = len(self.shape) - 1 - i
            for j, val in enumerate(row):
                if self.annoying:
                    a = 0
                else:
                    a = 1
                if val and grid[self.pos[1] + i + a][self.pos[0] + j].occ:
                    self.active = False
                    self.paint(grid)
                    return

    def paint(self, grid):
        """Copies the information of the block onto the affected cells"""
        for i, row in enumerate(self.shape):
            for j, val in enumerate(row):
                if val:
                    grid[self.pos[1] + i - self.annoying][self.pos[0] + j].color = self.color
                    grid[self.pos[1] + i - self.annoying][self.pos[0] + j].occ = True
