from typing import Optional

import numpy as np
import torch
from GameRules import YatzyGameState, Sheet


class Node:
    def __init__(self, game_state: YatzyGameState,
                 move: str = "",
                 prior: float = 0,
                 parent: Optional['Node'] = None
                 ) -> None:
        # game_state after move has been made
        self.game_state = game_state

        self.move: str = move
        self.parent: Optional['Node'] = parent

        self.children: list['Node'] = []

        self.value: float = 0
        self.average_value: float = 0
        self.visits: int = 0

        self.prior: float = prior
        self.evaluation: float

    def make_children(self, policy: torch.Tensor) -> None:
        moves = self.game_state.available_moves()
        for move in moves:
            state = self.game_state.copy()
            state.make_move(move)
            prior = policy[Sheet.move2index(move)].item()

            child = Node(state, move, prior, parent=self)

            if state.game_over():
                child.evaluation = state.get_status()

            self.children.append(child)

    def select_child(self, C: float) -> 'Node':
        if len(self.children) == 0:
            return self

        child_evaluations = np.zeros(len(self.children))
        for i, child in enumerate(self.children):
            child_evaluations[i] = self.calculate_PUCT(C, child)

        max_index = np.argmax(child_evaluations)
        return self.children[max_index]

    def calculate_PUCT(self, C: float, child: 'Node') -> float:
        relative_visits = np.sqrt(self.visits)/(child.visits+1)
        exploration = child.prior * relative_visits

        value = child.average_value + C * exploration
        return value

    def backpropagate(self) -> None:
        node = self
        while node is not None:
            node.visits += 1
            node.value += node.corresponding_sign(self.evaluation)
            node.average_value = node.value/node.visits
            node = node.parent

    def corresponding_sign(self, value: float) -> float:
        if self.my_player() == 1:
            return value
        return -value

    def my_player(self) -> int:
        if self.parent is None:
            return self.game_state.current_player
        return -self.game_state.current_player

    def choose_move(self) -> str:
        visits = [child.visits for child in self.children]
        max_visits = max(visits)
        max_index = visits.index(max_visits)

        chosen_child = self.children[max_index]
        return chosen_child.move
