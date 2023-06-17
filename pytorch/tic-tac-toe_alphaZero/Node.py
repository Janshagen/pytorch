import random
from typing import Optional

import numpy as np
import torch
from GameRules import TicTacToeGameState


class Node:
    def __init__(self, game_state: TicTacToeGameState,
                 move: Optional[tuple[int, int]] = None,
                 parent: Optional['Node'] = None) -> None:
        self.game_state = game_state

        self.move: Optional[tuple[int, int]] = move
        self.parent: Optional['Node'] = parent

        self.children: list['Node'] = []

        self.value: float = 0
        self.average_value: float = 0
        self.visits: int = 0

        self.prior: float
        self.evaluation: float

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

            relative_visits = np.sqrt(self.visits)/(child.visits+1)
            exploration = child.prior * relative_visits

            child_evaluations[i] = self.average_value + C * exploration

        max_index = np.argmax(child_evaluations)
        return self.children[max_index]

    def backpropagate(self) -> None:
        instance = self
        while instance is not None:
            instance.visits += 1
            instance.value += self.evaluation if instance.game_state.player == 1 \
                else -self.evaluation
            instance.average_value = instance.value/instance.visits
            instance = instance.parent

    def choose_move(self) -> tuple[int, int]:
        visits = [child.visits for child in self.children]
        max_visits = max(visits)
        max_index = visits.index(max_visits)

        chosen_child = self.children[max_index]
        assert chosen_child.move
        return chosen_child.move
