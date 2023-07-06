import random
import time

import numpy as np
import torch
from GameRules import TicTacToeGameState
from Node import Node
from TicTacToeModel import AlphaZero


class MCTS:
    def __init__(self, model: AlphaZero, exploration_rate: float,
                 sim_time: float = np.inf, sim_number: int = 1_000_000,
                 verbose: bool = False) -> None:

        self.exploration_rate = exploration_rate
        self.sim_time = sim_time
        self.sim_number = sim_number

        self.model = model
        self.root: Node

        self.verbose = verbose

    def find_move(self, game_state: TicTacToeGameState) -> tuple[int, int]:
        self.root = Node(game_state)

        start_time = time.process_time()
        for _ in range(self.sim_number):
            if self.maximum_time_exceeded(start_time):
                self.print_data_if_verbose()
                return self.root.choose_move()

            current = self.root
            current = self.traverse_tree(current)

            if current.visits > 0.5*self.sim_number:
                self.print_data_if_verbose()
                assert current.move
                return current.move

            if not current.game_state.game_over():
                current = self.expand_tree(current)

            current.backpropagate()

        self.print_data_if_verbose()
        return self.root.choose_move()

    def traverse_tree(self, current: Node) -> Node:
        current = current.select_child(self.exploration_rate)
        if current.visits > 0.5*self.sim_number:
            return current

        while len(current.children) > 0:
            current = current.select_child(self.exploration_rate)
        return current

    def expand_tree(self, current: Node) -> Node:
        torch_board = self.model.state2tensor(current.game_state)
        with torch.no_grad():
            evaluation, policy = self.model(torch_board)
            if current.parent is None:
                policy = self.model.add_noise(policy)

        current.evaluation = evaluation.item()
        current.make_children(policy[0])
        return current

    def print_data_if_verbose(self):
        if self.verbose:
            self.print_data()

    def maximum_time_exceeded(self, start_time: float) -> bool:
        return time.process_time() - start_time > self.sim_time

    def print_data(self) -> None:
        visits, val, p = self.root.visits, self.root.value, self.root.game_state.player
        print(
            f'root; player: {p}, rollouts: {visits}, value: {round(val, 2)}',
            f'vinstprocent: {round((visits+val)*50/visits, 2)}%')
        print('children;')
        print('visits:', end=' ')
        child_visits = [child.visits for child in self.root.children]
        child_values = [round(child.value, 2) for child in self.root.children]
        child_prior = [round(child.prior, 3) for child in self.root.children]
        print(child_visits)
        print('values:', end=' ')
        print(child_values)
        print('priors:', end=' ')
        print(child_prior)
        print('')

    def rollout(self, current: Node):
        state = current.game_state.copy()
        while True:
            moves = state.available_moves()
            state.make_move(random.choice(moves))

            if state.game_over():
                return state.get_status()
