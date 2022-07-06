import numpy as np
import torch

from gameplay import availableMoves, gameEnd, makeMove, nextPlayer, randomMove
from Node import Node
from TicTacToeModel import *


def AIfindMove(rootState: np.ndarray, rootPlayer: int, simulations: int, UCB1: float, model=None, device=None) -> int:
    moves = availableMoves(rootState)

    if not moves:
        return None

    root = Node(rootPlayer)
    root.makeChildren(rootPlayer, moves)

    for _ in range(simulations):
        currentState = rootState.copy()
        current = root

        # Tree traverse
        while len(current.children) > 0:
            current = current.selectChild(UCB1)
            makeMove(currentState, current.player, current.move)

            # returns a move if visits exceeds half of total simulations
            if current.visits >= 0.5*simulations:
                # printData(root)
                return current.move

        # Expand tree if current has been visited and isnt a terminal node
        if current.visits > 0 and not gameEnd(currentState).any():
            moves = availableMoves(currentState)
            current.makeChildren(current.nextPlayer(), moves)
            current = current.selectChild(UCB1)
            makeMove(currentState, current.player, current.move)

        # Rollout
        result = rollout(currentState, current.nextPlayer())
        # result = evaluation(currentState, model, device)

        # Backpropagation
        current.backpropagate(result)

    # printData(root)
    return root.chooseMove()


def rollout(currentState: np.ndarray, currentPlayer: int) -> np.ndarray:
    # finds a random move and executes it if possible. as long as gameEnd is False and movesInRollosut is less than cutoff
    while True:
        result = gameEnd(currentState)
        if result.any():
            return result

        moves = availableMoves(currentState)
        if not moves:
            return np.array([0, 0])

        move = randomMove(moves)
        makeMove(currentState, currentPlayer, move)
        currentPlayer = nextPlayer(currentPlayer)


def evaluation(board: np.ndarray, model, device) -> float:
    input = model.board2tensor(board, device)
    prob = model(input).item()

    return np.array([prob, -prob])


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
    HIDDEN_SIZE1 = HIDDEN_SIZE2 = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvModel(HIDDEN_SIZE1, HIDDEN_SIZE2, OUTPUT_SIZE).to(device)
    model.load_state_dict(torch.load(FILE))
    model.to(device)
    model.eval()

    return model, device


def printData(root: object) -> None:
    visits, val, p = root.visits, root.value, root.player
    print(
        f'root; player: {p}, rollouts: {visits}, value: {round(val*1, 2)}, vinstprocent: {round((visits+val)*50/visits, 2)}%')
    print('children;')
    print('visits:', end=' ')
    childVisits = [child.visits.tolist() for child in root.children]
    childVisits.sort(reverse=True)
    print(childVisits)
    print('')
