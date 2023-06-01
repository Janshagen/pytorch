import random
from typing import Optional

import numpy as np


class Node:
    def __init__(self, player: int, move: tuple[int, int] = (-1, -1),
                 parent: Optional['Node'] = None) -> None:
        self.value: float = 0
        self.visits: int = 0
        self.parent: Optional['Node'] = parent
        self.children: list['Node'] = []
        self.move: tuple[int, int] = move
        self.player: int = player      # self.player makes self.move

    def makeChildren(self, moves: list[tuple[int, int]]) -> None:
        """Makes a child node for every possible move"""
        player = self.nextPlayer()
        for move in moves:
            child = Node(player, move, parent=self)
            self.children.append(child)

        random.shuffle(self.children)

    def selectChild(self, C: float) -> 'Node':
        """Uses UCB1 to pick child node"""
        # if node doesn't have children, return self
        if len(self.children) == 0:
            return self

        UCB1values = np.zeros(len(self.children))
        for i, child in enumerate(self.children):
            # returns child if it hasn't been visited before
            if child.visits == 0:
                return child
            assert child.parent
            # calculates UCB1
            v = child.value
            mi = child.visits
            mp = child.parent.visits
            UCB1values[i] = v/mi + C * np.sqrt(np.log(mp)/mi)

        # return child that maximizes UCB1
        maxIndex = np.argmax(UCB1values)
        return self.children[maxIndex]

    def backpropagate(self, result: float) -> None:
        """Updates value and visits according to result"""
        instance = self
        while instance is not None:
            instance.visits += 1
            instance.value += result if instance.player == 1 else -result
            instance = instance.parent

    def chooseMove(self) -> tuple[int, int]:
        """Chooses most promising move from the list of children"""
        # if node doesn't have children, make no move
        if len(self.children) == 0:
            return self.move

        # finds child with most visits and returns it
        visits = [child.visits for child in self.children]
        maxVisits = max(visits)
        maxIndex = visits.index(maxVisits)

        chosenChild = self.children[maxIndex]
        return chosenChild.move

    def nextPlayer(self) -> int:
        return -1 if self.player == 1 else 1
