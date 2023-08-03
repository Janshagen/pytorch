import random

from fiveDimAlphaZero.fivedimMCTS import fiveMCTS
from fiveDimAlphaZero.fivedimTicTacToeModel import fiveAlphaZero
from fiveDimAlphaZero.fivedimGameRules import fiveTicTacToeGameState
from GameRules import TicTacToeGameState
from MCTS import MCTS
from TicTacToeModel import AlphaZero

EXPLORATION_RATE = 3
SIMULATIONS = 1000

NUMBER_OF_GAMES = 100

FILE_TWO_DIM = "AlphaZero2023-08-02 01:16.pth"
FILE_FIVE_DIM = "AlphaZero2023-07-30 14:22.pth"


def main():
    model_2dim = AlphaZero().load_model(FILE_TWO_DIM)
    model_5dim = fiveAlphaZero().load_model(FILE_FIVE_DIM)

    mcts_2dim = MCTS(model_2dim, EXPLORATION_RATE, sim_number=SIMULATIONS)
    mcts_5dim = fiveMCTS(model_5dim, EXPLORATION_RATE, sim_number=SIMULATIONS)

    print("Result as [2dim win, Draw, 5dim win]")
    accumulated_results = [0]*3
    for _ in range(NUMBER_OF_GAMES):
        result = game(mcts_2dim, mcts_5dim)
        accumulated_results[result+1] += 1
        print(accumulated_results)

    print("Result as [2dim win, Draw, 5dim win]")


def game(mcts_2dim: MCTS, mcts_5dim: fiveMCTS) -> int:
    starting_player = random.choice([1, -1])
    game_state_2dim = TicTacToeGameState.new_game(starting_player)
    game_state_5dim = fiveTicTacToeGameState.new_game(starting_player)

    game_history = []

    move = (10, 10)
    move_counter = 0
    while True:
        if game_state_2dim.player == -1:
            move = mcts_2dim.find_move(game_state_2dim)

        elif game_state_2dim.player == 1:
            move = mcts_5dim.find_move(game_state_5dim)

        if move_counter == 0:
            move = (random.choice([0, 1, 2]), random.choice([0, 1, 2]))
            move_counter += 1

        game_state_2dim.make_move(move)
        game_state_5dim.make_move(move)
        game_history.append(game_state_2dim.copy())

        if game_state_2dim.game_over():
            if game_state_5dim.get_status() != 0:
                for state in game_history:
                    print(state.board)
            return game_state_2dim.get_status()


if __name__ == '__main__':
    main()
