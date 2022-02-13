class Cell:
    def __init__(self, i, j, mine):
        self.i = i
        self.j = j
        self.mine = mine
        self.revealed = False
        self.neighbor_count = 0
        self.picked = False

    def neighbor(self, grid):
        """Räknar antalet grannar som är minor"""
        total = 0
        for n in range(-1, 2):
            for k in range(-1, 2):
                if grid[self.i + n][self.j + k].mine:
                    total += 1
        return total
