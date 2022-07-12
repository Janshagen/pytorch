import numpy as np
import torch
import torch.nn as nn

from Connect4Model import Model
from gameplay import availableMoves, gameEnd, makeMove, nextPlayer, randomMove
from Node import Node


def MCTSfindMove(rootState: np.ndarray, rootPlayer: int, simulations: int, UCB1: float, model=None, device=None) -> int:
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
        result = rollout(currentState, current.nextPlayer(), model, device)

        # Backpropagation
        current.backpropagate(result)

    # printData(root)
    return root.chooseMove()


def rollout(currentState: np.ndarray, currentPlayer: int, model: nn.Module, device: torch.device) -> np.ndarray:
    # finds a random move and executes it if possible
    while True:
        result = gameEnd(currentState)
        if result.any():
            return result

        # return evaluation(currentState, currentPlayer, model, device)

        moves = availableMoves(currentState)
        if not moves:
            return np.array([0, 0])

        move = randomMove(moves)
        makeMove(currentState, currentPlayer, move)
        currentPlayer = nextPlayer(currentPlayer)


def evaluation(board: np.ndarray, currentPlayer: int, model: nn.Module, device: torch.device) -> np.ndarray:
    input = model.board2tensor(board, currentPlayer, device)
    prob = model(input)[0][0]
    prob = torch.softmax(prob, dim=0)
    eval = currentPlayer * (prob[0]-prob[2]).item()
    return np.array([eval, -eval])


def loadModel():
    FILE = '/home/anton/skola/egen/pytorch/connect4/Connect4model.pth'
    OUTPUT_SIZE = 3
    HIDDEN_SIZE1 = HIDDEN_SIZE2 = 72
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(HIDDEN_SIZE1, HIDDEN_SIZE2, OUTPUT_SIZE).to(device)
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
