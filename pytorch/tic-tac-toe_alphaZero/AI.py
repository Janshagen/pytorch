from typing import Optional

import numpy as np
import numpy.typing as npt
import torch
from gameplay import availableMoves, gameEnd, makeMove
from MCTSData import MCTSData
from Node import Node
from TicTacToeModel import ConvModel, LinearModel

board_type = npt.NDArray[np.int8]


def MCTSfindMove(data: MCTSData) -> Optional[tuple[int, int]]:
    moves = availableMoves(data.board)

    if not moves:
        return None

    root = Node(data.player)
    root.makeChildren(moves)

    for _ in range(data.sim_number):
        current_board = data.board.copy()
        current = root

        # Tree traverse
        while len(current.children) > 0:
            current = current.selectChild(data.UCB1)
            makeMove(current_board, current.player, current.move)

            # returns a move if visits exceeds half of total simulations
            if current.visits >= 0.5*data.sim_number:
                # printData(root)
                return current.move

        # Expand tree if current has been visited and isn't a terminal node
        if current.visits > 0 and not gameEnd(current_board):
            moves = availableMoves(current_board)
            current.makeChildren(moves)
            current = current.selectChild(data.UCB1)
            makeMove(current_board, current.player, current.move)

        # Rollout/Evaluation
        # evaluation = evaluationLinear(data, current_board,
        # current.nextPlayer())
        evaluation = evaluationConv(data, current_board, current.nextPlayer())

        # Backpropagation
        current.backpropagate(evaluation)

    # printData(root)
    return root.chooseMove()


def evaluationLinear(data: MCTSData, board: board_type,
                     player: int) -> float:
    prob = data.model(board, player, data.device).item()
    # prob - (1-prob) = 2*prob-1
    prob = 2*prob-1
    return prob


def evaluationConv(data: MCTSData, board: board_type,
                   player: int) -> float:
    prob = data.model(board, player, data.device)[0][0]
    prob = torch.softmax(prob, dim=0)
    eval = player * (prob[0]-prob[2]).item()
    return eval


def bestEvaluationFindMove(data: MCTSData) -> tuple[int, int]:
    """Chooses move which maximizes players evaluation, and minimizes
    opponents evaluation, of gamestate after that move is made."""
    moves = availableMoves(data.board)
    evaluations = np.empty(len(moves))
    for i, move in enumerate(moves):
        board_ = data.board.copy()
        makeMove(board_, data.player, move)
        eval = data.model(board_, data.player, data.device)[0][0]
        eval = torch.softmax(eval, dim=0)
        evaluations[i] = data.player * (eval[0] - eval[2]).item()
    maxIndex = np.argmax(evaluations)

    return moves[maxIndex]


def loadLinearModel():
    FILE = '/home/anton/skola/egen/pytorch/tic-tac-toe/TicTacToeModel.pth'
    INPUT_SIZE = 18
    OUTPUT_SIZE = 1
    HIDDEN_SIZE1 = HIDDEN_SIZE2 = HIDDEN_SIZE3 = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LinearModel(INPUT_SIZE, HIDDEN_SIZE1,
                        HIDDEN_SIZE2, HIDDEN_SIZE3, OUTPUT_SIZE).to(device)
    model.load_state_dict(torch.load(FILE))
    model.to(device)
    model.eval()

    return model, device


def loadConvModel():
    FILE = '/home/anton/skola/egen/pytorch/tic-tac-toe/TicTacToeModelConv.pth'
    OUTPUT_SIZE = 3
    HIDDEN_SIZE1 = HIDDEN_SIZE2 = 72
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvModel(HIDDEN_SIZE1, HIDDEN_SIZE2, OUTPUT_SIZE).to(device)
    model.load_state_dict(torch.load(FILE))
    model.to(device)
    model.eval()

    return model, device


def printData(root: Node) -> None:
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
