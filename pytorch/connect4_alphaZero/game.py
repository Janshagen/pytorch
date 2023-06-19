import pygame
from MCTS import MCTS
from GameRules import Connect4GameState
from Connect4Model import AlphaZero
from interface import (choose_config, draw, game_over, initialize_game,
                       resolve_event)


# Configurations
SIMULATIONS = 300
WIDTH = 120
HEIGHT = int(WIDTH*0.8)
EXPLORATION_RATE = 1.4


def main() -> None:
    sims = choose_config(SIMULATIONS)
    screen, frame = initialize_game(WIDTH, HEIGHT)

    game_state = Connect4GameState.new_game(starting_player=-1)
    model = AlphaZero().load_model()
    mcts = MCTS(model, EXPLORATION_RATE, sim_number=sims, verbose=True)

    draw(screen, frame, game_state,  WIDTH, HEIGHT)

    result = game(mcts, game_state,  screen, frame)
    if not game_over(screen, result, WIDTH):
        print("###### NEW GAME ######")
        main()


def game(mcts: MCTS, game_state: Connect4GameState,
         screen: pygame.surface.Surface, frame: pygame.Surface) -> int:
    while True:
        move = None
        # Human
        if game_state.player == -1:
            move = resolve_event(game_state, WIDTH)
            if move:
                game_state.make_move(move)

        # AI
        elif game_state.player == 1:
            move = mcts.find_move(game_state)
            game_state.make_move(move)
            resolve_event(game_state, WIDTH)

            print_data(game_state, mcts.model)

        draw(screen, frame, game_state, WIDTH, HEIGHT, move)

        if game_state.game_over():
            return game_state.get_status()


def print_data(game_state: Connect4GameState, model: AlphaZero) -> None:
    torch_board = model.state2tensor(game_state)
    print(f"evaluation:{model(torch_board)[0][0][0].item():.4f}")
    print(f"policy: {model(torch_board)[1][0]}")


if __name__ == '__main__':
    main()
