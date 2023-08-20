import random
from Connect4Model import AlphaZero
from GameRules import Connect4GameState
from MCTS import MCTS
import matplotlib.pyplot as plt

EXPLORATION_RATE = 4
SIMULATIONS = 250
TIME = 1000  # second

FILE_A = "AlphaZero2023-07-03 18:55.pth"
FILE_B = "AlphaZero2023-06-22 03:47.pth"

NUMBER_OF_GAMES = 10


def main():
    model_A = AlphaZero().load_model()
    model_A.eval()
    model_B = AlphaZero().load_model()
    model_B.eval()

    alpha_zero = MCTS(model_A, EXPLORATION_RATE, sim_time=TIME, sim_number=SIMULATIONS)
    rollout = MCTS(None, EXPLORATION_RATE, sim_time=TIME, sim_number=SIMULATIONS)

    print("Result as [alpha_zero win, Draw, rollout win]")
    accumulated_results = [0]*3
    for _ in range(NUMBER_OF_GAMES):
        result, evaluations = game(alpha_zero, rollout)
        accumulated_results[result+1] += 1
        print(accumulated_results)

        color = 'g' if result == -1 else 'r'
        plt.plot(evaluations, color)

    print("Result as [alpha_zero win, Draw, rollout win]")
    print(evaluations)
    plt.show()


def game(mcts_A: MCTS, mcts_B: MCTS) -> int:
    evaluations = []
    game_state = Connect4GameState.new_game(random.choice([1, -1]))
    move = 0
    while True:
        if game_state.player == -1:
            move = mcts_A.find_move(game_state)

        elif game_state.player == 1:
            move = mcts_B.find_move(game_state)

        game_state.make_move(move)
        evaluations.append(-mcts_A.model(mcts_A.model.state2tensor(game_state))
                           [0][0][0].cpu().item() * game_state.player)

        if game_state.game_over():
            print(game_state.board)
            return game_state.get_status(), evaluations


if __name__ == '__main__':
    main()
