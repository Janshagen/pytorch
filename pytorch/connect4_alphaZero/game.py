from MCTS import MCTS
from GameRules import Connect4GameState
from Connect4Model import AlphaZero
from interface import InterfaceConnect4


# Configurations
SIMULATIONS = 300
WIDTH = 120
HEIGHT = int(WIDTH*0.8)
EXPLORATION_RATE = 1.4


def main() -> None:
    interface = InterfaceConnect4(HEIGHT, WIDTH)
    simulations = interface.choose_config(SIMULATIONS)

    game_state = Connect4GameState.new_game(starting_player=-1)
    model = AlphaZero().load_model()

    mcts = MCTS(model, EXPLORATION_RATE, sim_number=simulations, verbose=True)

    interface.draw(game_state)
    result = game(mcts, game_state, interface)
    if interface.play_again(result):
        print("###### NEW GAME ######")
        main()


def game(mcts: MCTS, game_state: Connect4GameState, interface: InterfaceConnect4) -> int:
    while True:
        move = None
        # Human
        if game_state.player == -1:
            move = interface.resolve_event(game_state)
            if move:
                game_state.make_move(move)

        # AI
        elif game_state.player == 1:
            move = mcts.find_move(game_state)
            game_state.make_move(move)
            interface.resolve_event(game_state)

            print_data(game_state, mcts.model)

        interface.draw(game_state, move)

        if game_state.game_over():
            return game_state.get_status()


def print_data(game_state: Connect4GameState, model: AlphaZero) -> None:
    torch_board = model.state2tensor(game_state)
    print(f"evaluation:{model(torch_board)[0][0][0].item():.4f}")
    print(f"policy: {model(torch_board)[1][0]}")


if __name__ == '__main__':
    main()
