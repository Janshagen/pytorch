from typing import Optional
import re
import numpy as np
import torch
from GameRules import TicTacToeGameState


class Node:
    def __init__(self, game_state: TicTacToeGameState,
                 move: Optional[tuple[int, int]] = None,
                 prior: float = 0,
                 parent: Optional['Node'] = None
                 ) -> None:
        # game_state after move has been made
        self.game_state = game_state

        self.move: Optional[tuple[int, int]] = move
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
            prior = policy[TicTacToeGameState.move2index(move)].item()

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
        evaluation = self.evaluation
        node = self
        while node.parent is not None:
            node.visits += 1
            # change to every other
            node.value -= evaluation
            evaluation *= -1
            node.average_value = node.value/node.visits
            node = node.parent

        # for root node
        node.visits += 1
        node.value += evaluation

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

    def choose_move(self) -> tuple[int, int]:
        visits = [child.visits for child in self.children]
        max_visits = max(visits)
        max_index = visits.index(max_visits)

        chosen_child = self.children[max_index]
        assert chosen_child.move
        return chosen_child.move

    def print_tree(self, max_level: int):
        data = self.tree_traverse(max_level)
        for d in data:
            if d == "first":
                print(f"(, {self.visits})", end="\n{[")
            elif d == "new generation":
                print("]}\n{[", end="")
            elif d == "child of new parent":
                print("]  [", end="")
            else:
                print(d, end="")
        print("\n")

    def tree_traverse(self, max_level: int) -> list[str]:
        to_be_traversed: list[tuple['Node', int, str]] = [(self, 0, " ")]
        instructions = ["first"]
        previous_level = 0
        while to_be_traversed and previous_level < max_level:
            node, level, name_of_node = to_be_traversed.pop(0)

            previous_level = \
                self.check_if_new_generation(instructions, previous_level, level)
            self.check_if_child_of_new_parent(instructions, name_of_node)
            self.add_new_children(to_be_traversed, node, level, name_of_node)

            if node.visits != 0:
                instructions.append(
                    # CHANGE WHAT TO PRINT HERE
                    f"({name_of_node}, {node.evaluation:.2f}, {node.value:.2f})")

        instructions.pop(1)
        instructions.pop(1)
        instructions.append("]}")
        return instructions

    def check_if_new_generation(self, instructions: list[str],
                                previous_level: int,
                                level: int):
        if previous_level != level:
            if instructions[-1] == "child of new parent":
                instructions.pop()
            instructions.append("new generation")
        return level

    def check_if_child_of_new_parent(self, instructions: list[str],
                                     name_of_node: str):
        node_name = r"\((.*?),"
        match = re.search(node_name, instructions[-1])
        if match and len(match.group(1)) >= 3 and len(name_of_node) >= 3:
            if match.group(1)[-3] != name_of_node[-3]:
                instructions.append("child of new parent")

    def add_new_children(self, to_be_traversed: list,
                         node: 'Node',
                         level: int,
                         name_of_node: str):
        for child in node.children:
            child_name = f"{name_of_node}:{child.move}"
            if level == 0:
                child_name = f"{child.move}"

            to_be_traversed.append((child, level + 1, child_name))

    # () = one node
    # [] = children of one parent
    # {} = one generation
    # Node is only shown if visits > 0
