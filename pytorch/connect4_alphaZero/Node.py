from typing import Optional

import numpy as np
import torch
from GameRules import Connect4GameState


class Node:
    def __init__(self, game_state: Connect4GameState,
                 move: int = 10,
                 prior: float = 0,
                 parent: Optional['Node'] = None
                 ) -> None:
        # game_state after move has been made
        self.game_state = game_state

        self.move: int = move
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
            prior = policy[move].item()

            child = Node(state, move, prior, parent=self)
            self.children.append(child)

    def select_child(self, C: float) -> 'Node':
        if not self.has_children():
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
            return self.game_state.player
        return -self.game_state.player

    def has_children(self) -> bool:
        return len(self.children) > 0

    def choose_move(self) -> int:
        visits = [child.visits for child in self.children]
        max_visits = max(visits)
        max_index = visits.index(max_visits)

        chosen_child = self.children[max_index]
        return chosen_child.move

    def print_tree(self, model):
        data = self.tree_traverse(model)
        for d in data:
            if d == "first":
                print(f"(, {self.visits})", end="\n{")
            elif d == "new generation":
                print("]}\n{[", end="")
            elif d == "new child of same parent":
                print("]  [", end="")
            else:
                print(d, end="")
        print("\n")

    def tree_traverse(self, model) -> list[str]:
        to_be_traversed: list[tuple['Node', int, str]] = [(self, 0, " ")]
        instructions = ["first"]
        previous_level = 0
        while to_be_traversed:
            node, level, name_of_node = to_be_traversed.pop(0)

            if previous_level != level:
                instructions.append("new generation")
            previous_level = level

            if name_of_node[-1] == "0" and instructions[-1] != "new generation":
                instructions.append("new child of same parent")

            for i, child in enumerate(node.children):
                child_name = f"{name_of_node}:{i}"
                if instructions[-1] == "first":
                    child_name = f"{i}"

                to_be_traversed.append((child, level + 1, child_name))

            if node.visits != 0:
                # torch_board = model.state2tensor(current.game_state)
                # evaluation, _ = model.forward(torch_board)
                instructions.append(f"({name_of_node}, {node.visits})")

        instructions.pop(1)
        instructions.pop(1)
        instructions.append("]}")
        return instructions
