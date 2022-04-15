import numpy as np
import random


class Node:
    def __init__(self, num, move=None, parent=None) -> None:
        self.value = np.array(0, np.float32)
        self.visits = np.array(0, np.int32)
        self.parent = parent
        self.children = []
        self.move = move
        self.num = num                    # self.player makes self.move

    def makeChildren(self, player, moves) -> None:
        """Makes a child node for every possible move"""
        for move in moves:
            child = Node(player, move, parent=self)
            self.children.append(child)

        random.shuffle(self.children)

    def selectChild(self) -> object:
        """Uses UCB1 to pick child node"""
        # if node doesnt have children, return self
        if len(self.children) == 0:
            return self

        UCB1values = np.zeros(len(self.children))
        for i, child in enumerate(self.children):
            # returns child if it hasnt been visited before
            if child.visits == 0:
                return child

            # calculates UCB1
            v = child.value
            mi = child.visits
            mp = child.parent.visits
            UCB1values[i] = v/mi + 1 * np.sqrt(np.log(mp)/mi)

        # return child that maximises UCB1
        maxIndex = np.argmax(UCB1values)
        return self.children[maxIndex]

    def backpropagate(self, result) -> None:
        """Updates value and visits according to result"""
        instance = self
        while instance != None:
            instance.visits += 1
            instance.value += result[instance.num - 1]
            instance = instance.parent

    def chooseMove(self) -> np.ndarray:
        """Chooses most promising move from the list of children"""
        # if node doesnt have children, make no move
        if len(self.children) == 0:
            return self.move

        # finds child with most visits and returns it
        visits = [child.visits for child in self.children]
        maxVisits = max(visits)
        maxIndex = visits.index(maxVisits)

        chosenChild = self.children[maxIndex]
        return chosenChild.move

    def nextPlayer(self) -> int:
        return (self.num + 2) % 2 + 1
