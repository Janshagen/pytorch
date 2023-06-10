from typing import Optional

import numpy as np
import numpy.typing as npt
import torch
from gameplay import (available_moves, make_move, next_player)
from MCTSData import MCTSData
from Node import Node
from TicTacToeModel import AlphaZero

board_type = npt.NDArray[np.int8]


def MCTS_find_move(data: MCTSData) -> Optional[tuple[int, int]]:
    evaluation, policy = data.model(data.board, data.player)
    data.root = Node(data.board, data.player)

    # data.root = Node(data.board, next_player(data.player))
    # data.root.make_children(policy)
    # data.root.player = data.root.next_player()

    for _ in range(data.sim_number):
        current = data.root

        # Tree traverse
        while len(current.children) > 0:
            current = current.select_child(data.UCB1)

            # returns a move if visits exceeds half of total simulations
            if current.visits >= 0.5*data.sim_number:
                # printData(root)
                return current.move

        # Expand tree if it isn't a terminal node
        if not current.terminal_node:
            evaluation, policy = data.model(
                current.board, current.next_player())
            current.evaluation = evaluation

            current.make_children(policy[0])
            current = current.select_child(data.UCB1)

        # Backpropagation
        current.backpropagate()

    # printData(root)
    return data.root.choose_move()


def best_evaluation_find_move(data: MCTSData) -> tuple[int, int]:
    """Chooses move which maximizes players evaluation, and minimizes
    opponents evaluation, of gamestate after that move is made."""
    moves = available_moves(data.board)
    evaluations = np.empty(len(moves))
    for i, move in enumerate(moves):
        board = data.board.copy()
        make_move(board, data.player, move)
        eval = data.model(board, data.device, next_player(data.player))[0][0]
        eval = torch.softmax(eval, dim=0)
        evaluations[i] = (eval[0] - eval[2]).item()
    maxIndex = np.argmax(evaluations)

    return moves[maxIndex]


def load_model():
    FILE = '/home/anton/skola/egen/pytorch/tic-tac-toe/alpha_zero.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AlphaZero(device)
    model.load_state_dict(torch.load(FILE))
    model.to(device)
    model.eval()

    return model, device


def print_data(root: Node) -> None:
    visits, val, p = root.visits, root.value, root.player
    print(
        f'root; player: {p}, rollouts: {visits}, value: {round(val*1, 2)}',
        f'vinstprocent: {round((visits+val)*50/visits, 2)}%')
    print('children;')
    print('visits:', end=' ')
    childVisits = [child.visits for child in root.children]
    childVisits.sort(reverse=True)
    print(childVisits)
    print('')
