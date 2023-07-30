import pygame
from MCTS import MCTS
from GameRules import TicTacToeGameState
from TicTacToeModel import AlphaZero
from interface import (chooseConfig, draw, gameOver, initializeGame,
                       resolveEvent)


# Configurations
SIMULATIONS = 1000
WIDTH = 200
EXPLORATION_RATE = 3


def main() -> None:
    sims = chooseConfig(SIMULATIONS)
    screen, frame = initializeGame(WIDTH)

    game_state = TicTacToeGameState.new_game(starting_player=-1)
    model = AlphaZero().load_model()
    mcts = MCTS(model, EXPLORATION_RATE, sim_number=sims, verbose=True)

    draw(screen, frame, game_state, WIDTH)

    result = game(mcts, game_state,  screen, frame)
    if not gameOver(screen, result, WIDTH):
        print("###### NEW GAME ######")
        main()


def game(mcts: MCTS, game_state: TicTacToeGameState,
         screen: pygame.surface.Surface, frame: pygame.Surface) -> int:
    move_number = 0
    while True:
        # Human
        if game_state.player == -1:
            move = resolveEvent(game_state, WIDTH)
            if move:
                game_state.make_move(move)

        # AI
        elif game_state.player == 1:
            move_number += 1
            print(f"Move number: {move_number}")

            move = mcts.find_move(game_state)
            game_state.make_move(move)

            print_data(game_state, mcts.model)

        draw(screen, frame, game_state, WIDTH)

        if game_state.game_over():
            return game_state.get_status()


def print_data(game_state: TicTacToeGameState, model: AlphaZero) -> None:
    torch_board = model.state2tensor(game_state)
    print(f"Evaluation after move: {model(torch_board)[0][0][0].item():.4f}")
    print(" ")


if __name__ == '__main__':
    main()
