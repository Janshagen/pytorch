import numpy as np
from gameplay import availableMoves, gameEnd, makeMove, nextPlayer, randomMove
from Node import Node


def AIfindMove(rootState: np.ndarray, rootPlayer: int, simulations: int, UCB1: float) -> int:
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
                printData(root)
                return current.move

        # Expand tree if current has been visited and isnt a terminal node
        if current.visits > 0 and not gameEnd(currentState).any():
            moves = availableMoves(currentState)
            current.makeChildren(current.nextPlayer(), moves)
            current = current.selectChild(UCB1)
            makeMove(currentState, current.player, current.move)

        # Rollout
        result = rollout(currentState, current.nextPlayer())

        # Backpropagation
        current.backpropagate(result)

    printData(root)
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
