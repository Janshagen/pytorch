from MCTS import MCTS
from GameRules import Connect4GameState
from Connect4Model import AlphaZero
from interface import InterfaceConnect4


# Configurations
SIMULATIONS = 1000
WIDTH = 120
HEIGHT = int(WIDTH*0.8)
EXPLORATION_RATE = 5


def main() -> None:
    interface = InterfaceConnect4(WIDTH, HEIGHT)
    simulations = interface.choose_config(SIMULATIONS)

    game_state = Connect4GameState.new_game(starting_player=-1)
    model = AlphaZero().load_model("AlphaZero2023-07-23 11:47.pth")
    model.eval()

    mcts = MCTS(model, EXPLORATION_RATE, sim_number=simulations, verbose=True)

    interface.draw(game_state)
    result = game(mcts, game_state, interface)
    if interface.play_again(result):
        print("###### NEW GAME ######")
        main()


def game(mcts: MCTS, game_state: Connect4GameState, interface: InterfaceConnect4) -> int:
    move_number = 0
    while True:
        move = None
        # Human
        if game_state.player == 1:
            move = interface.resolve_event(game_state)
            if move is not None:
                game_state.make_move(move)

        # AI
        elif game_state.player == -1:
            move_number += 1
            print(f"Move number: {move_number}")

            move = mcts.find_move(game_state)
            game_state.make_move(move)

            print_data(game_state, mcts.model)

        interface.draw(game_state, move)

        if game_state.game_over():
            return game_state.get_status()


def print_data(game_state: Connect4GameState, model: AlphaZero) -> None:
    torch_board = model.state2tensor(game_state)
    print(f"Evaluation after move: {model(torch_board)[0][0][0].item():.4f}")
    print(" ")


if __name__ == '__main__':
    main()
