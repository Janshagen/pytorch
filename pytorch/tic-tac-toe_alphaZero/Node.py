import random
from typing import Optional

import numpy as np
import torch
from GameRules import TicTacToeGameState


class Node:
    def __init__(self, game_state: TicTacToeGameState,
                 move: Optional[tuple[int, int]] = None,
                 parent: Optional['Node'] = None) -> None:
        self.value: float = 0
        self.average_value: float = 0
        self.visits: int = 0
        self.parent: Optional['Node'] = parent
        self.children: list['Node'] = []

        # self.player makes self.move
        # board after move has been made
        self.move: Optional[tuple[int, int]] = move
        self.game_state = game_state

        self.evaluation: float
        self.prior: float
        self.terminal_node: bool = False

    def make_children(self, policy: torch.Tensor) -> None:
        moves = self.game_state.available_moves()
        for move in moves:
            state = self.game_state.copy()
            state.make_move(move)

            child = Node(state, move, parent=self)
            move_index = TicTacToeGameState.move2index(move)
            child.prior = policy[move_index].item()

            status = state.game_status()
            if TicTacToeGameState.game_over(status):
                self.terminal_node = True
                child.evaluation = status

            self.children.append(child)

        random.shuffle(self.children)

    def select_child(self, C: float) -> 'Node':
        if len(self.children) == 0:
            return self

        child_evaluations = np.zeros(len(self.children))
        for i, child in enumerate(self.children):
            assert child.parent

            relative_visits = np.sqrt(child.parent.visits)/(child.visits+1)
            exploration = child.prior * relative_visits

            child_evaluations[i] = self.average_value + C * exploration

        maxIndex = np.argmax(child_evaluations)
        return self.children[maxIndex]

    def backpropagate(self) -> None:
        assert self.parent
        instance = self
        while instance is not None:
            instance.visits += 1
            instance.value += self.parent.evaluation if instance.game_state.player == 1 \
                else -self.parent.evaluation
            instance.average_value = instance.value/instance.visits
            instance = instance.parent

    def choose_move(self) -> tuple[int, int]:
        visits = [child.visits for child in self.children]
        maxVisits = max(visits)
        maxIndex = visits.index(maxVisits)

        chosenChild = self.children[maxIndex]
        assert chosenChild.move
        return chosenChild.move
