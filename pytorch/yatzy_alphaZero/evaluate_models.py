import random
from YatzyModel import AlphaZero
from GameRules import YatzyGameState
from MCTS import MCTS

EXPLORATION_RATE = 1.4
SIMULATIONS = 100

FILE_A = "AlphaZero2023-06-18 18:51.pth"
FILE_B = "AlphaZero2023-06-18 18:51.pth"


def main():
    model_A = AlphaZero().load_model(FILE_A)
    model_A.eval()
    model_B = AlphaZero().load_model(FILE_B)
    model_B.eval()

    mcts_A = MCTS(model_A, EXPLORATION_RATE, sim_number=SIMULATIONS)
    mcts_B = MCTS(model_B, EXPLORATION_RATE, sim_number=SIMULATIONS)

    accumulated_results = [0]*3
    for _ in range(5):
        result = game(mcts_A, mcts_B)
        accumulated_results[result+1] += 1

    print("Result as [A win, Draw, B win]")
    print(accumulated_results)


def game(mcts_A: MCTS, mcts_B: MCTS) -> int:
    game_state = YatzyGameState.new_game(random.choice([1, -1]))
    move = ""
    while True:
        if game_state.current_player == -1:
            move = mcts_A.find_move(game_state)

        elif game_state.current_player == 1:
            move = mcts_B.find_move(game_state)

        game_state.make_move(move)

        if game_state.game_over():
            return game_state.get_status()


if __name__ == '__main__':
    main()
