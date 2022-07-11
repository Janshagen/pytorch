import numpy as np
import torch

from gameplay import availableMoves, gameEnd, makeMove, nextPlayer, randomMove
from Node import Node
from TicTacToeModel import *


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
        result = rollout(currentState, current.nextPlayer())

        # result = evaluation(currentState, model, device)
        # result = evaluationConv(currentState, current.nextPlayer(), model, device)

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


def evaluation(board: np.ndarray, model: nn.Module, device: torch.device) -> np.ndarray:
    input = model.board2tensor(board, device)
    prob = model(input).item()

    return np.array([prob, 1-prob])


def evaluationConv(board: np.ndarray, player: int, model: nn.Module, device: torch.device) -> np.ndarray:
    input = model.board2tensor(board, player, device)
    prob = model(input)
    # playerIndex = 0 if player == 1 else 2
    prob = torch.softmax(prob, 2)[0][0]

    # evntuellt olika probabilities här
    return np.array([prob[0].item(), prob[2].item()])


def bestEvaluationFindMove(board: np.ndarray, player: int, model: nn.Module, device: torch.device):
    moves = availableMoves(board)
    evaluations = np.empty(len(moves))
    for i, move in enumerate(moves):
        board_ = board.copy()
        makeMove(board_, player, move)
        evaluations[i] = model(model.board2tensor(
            board_, nextPlayer(player), device))[0][0][0].item()
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
