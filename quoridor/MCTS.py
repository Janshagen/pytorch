from Node import Node
from gameplay import availableMoves, randomMove, makeMove, gameEnd, winner, locatePlayer
import numpy as np
import time


def AIfindMove(rootState: np.ndarray, players, sims: int, THINK_TIME: int, UCB1: float, CUTOFF: int, MATRIX_SIZE) -> np.ndarray:
    rootPlayer = players[0]
    root = Node(rootPlayer.num)
    moves = availableMoves(rootState, players, rootPlayer.num, MATRIX_SIZE)
    root.makeChildren(rootPlayer.num, moves)

    startTime = time.process_time()
    i = 0
    for _ in range(sims):
        # if time.process_time() - startTime > THINK_TIME:
        #     printData(root)
        #     print(time.process_time() - startTime)
        #     return root.chooseMove()
        currentState = rootState.copy()
        current = root

        # Tree traverse
        while len(current.children) > 0:
            current = current.selectChild(UCB1)
            makeMove(currentState, current.move, current.num)

            # returns a move if visits exceeds half of total simulations
            if current.visits >= 0.5*sims:
                printData(root)
                return current.move

        # Expand tree if current hasnt been visited and isnt a terminal node
        if current.visits > 0 and not gameEnd(players, current.num, current.move):
            moves = availableMoves(currentState, players,
                                   current.nextPlayer(), MATRIX_SIZE)
            current.makeChildren(current.nextPlayer(), moves)
            current = current.selectChild(UCB1)
            makeMove(currentState, current.move, current.num)

        # Rollout
        # result = rollout(currentState, current.num, players,
        #                  current.move, CUTOFF, MATRIX_SIZE)
        result = evaluation(currentState, players)

        # Backpropagation
        current.backpropagate(result)

    return root.chooseMove()


def rollout(currentState: np.ndarray, currentPlayer: int, players, move: tuple, CUTOFF, MATRIX_SIZE) -> np.ndarray:
    # finds a random move and executes it if possible. as long as gameEnd is False and movesInRollosut is less than cutoff
    movesInRollout = 0
    while True:
        if gameEnd(players, currentPlayer, move):
            return winner(currentPlayer)

        if movesInRollout >= CUTOFF:
            return evaluation(currentState, players)
        currentPlayer = (currentPlayer + 2) % 2 + 1

        moves = availableMoves(currentState, players,
                               currentPlayer, MATRIX_SIZE)
        move = randomMove(moves)
        makeMove(currentState, move, currentPlayer)
        movesInRollout += 1


def evaluation(board, players):
    result = np.zeros(2)
    for player in players:
        pos = locatePlayer(board, player.num)
        result[player.num-1] = player.distanceToGoal(board, pos)

    prob = (result[1]-result[0])/result.sum()
    return np.array([prob, -prob])


def printData(root: object) -> None:
    r, val, p = root.visits, root.value, root.num
    print(
        f'root; player: {p}, rollouts: {r}, value: {np.around(val, 1)}, vinstprocent: {round((r+val)*100/2/r, 2)}%')
    print('children;')
    print('visits, value:', end=' ')
    childVisits = [(child.visits.tolist(), child.value.tolist())
                   for child in root.children]
    childVisits.sort(reverse=True)
    print(childVisits)
    print('')
