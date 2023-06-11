import time

import numpy as np
from DeepLearningData import DeepLearningData
from Node import Node

from GameRules import TicTacToeGameState


class MCTS:
    def __init__(self, UCB1: float, sim_time: float = np.inf,
                 sim_number: int = 1_000_000, cutoff: int = 0) -> None:

        self.UCB1 = UCB1
        self.sim_time = sim_time
        self.sim_number = sim_number
        self.cutoff = cutoff

        self.root: Node

    def find_move(self, game_state: TicTacToeGameState,
                  learning_data: DeepLearningData) -> tuple[int, int]:
        self.root = Node(game_state)

        start_time = time.process_time()
        for _ in range(self.sim_number):
            if self.maximum_time_exceeded(start_time):
                return self.root.choose_move()

            current = self.root
            current = self.traverse_tree(current)

            if current.visits >= 0.5*self.sim_number:
                # print_data(root)
                assert current.move
                return current.move

            if not current.terminal_node:
                current = self.expand_tree(learning_data, current)

            current.backpropagate()

        # print_data(root)
        return self.root.choose_move()

    def traverse_tree(self, current: Node) -> Node:
        while len(current.children) > 0:
            current = current.select_child(self.UCB1)
            if current.visits >= 0.5*self.sim_number:
                return current
        return current

    def expand_tree(self, learning_data: DeepLearningData, current: Node) -> Node:
        torch_board = learning_data.model.state2tensor(current.game_state)
        evaluation, policy = learning_data.model(torch_board)
        current.evaluation = evaluation

        current.make_children(policy[0])
        current = current.select_child(self.UCB1)
        return current

    def maximum_time_exceeded(self, start_time: float) -> bool:
        return time.process_time() - start_time > self.sim_time

    def print_data(self) -> None:
        visits, val, p = self.root.visits, self.root.value, self.root.game_state.player
        print(
            f'root; player: {p}, rollouts: {visits}, value: {round(val*1, 2)}',
            f'vinstprocent: {round((visits+val)*50/visits, 2)}%')
        print('children;')
        print('visits:', end=' ')
        childVisits = [child.visits for child in self.root.children]
        childVisits.sort(reverse=True)
        print(childVisits)
        print('')
