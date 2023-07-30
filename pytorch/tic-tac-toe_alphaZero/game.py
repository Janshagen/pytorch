import pygame
from MCTS import MCTS
from GameRules import TicTacToeGameState
from TicTacToeModel import AlphaZero
from interface import (chooseConfig, draw, gameOver, initializeGame,
                       resolveEvent)
import torch


# Configurations
SIMULATIONS = 50
WIDTH = 200
EXPLORATION_RATE = 3


def main() -> None:
    sims = chooseConfig(SIMULATIONS)
    screen, frame = initializeGame(WIDTH)

    game_state = TicTacToeGameState.new_game(starting_player=1)
    model = AlphaZero().load_model("AlphaZero2023-07-30 14:22.pth")
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


def best_evaluation_find_move(game_state: TicTacToeGameState, model: AlphaZero):
    moves = game_state.available_moves()
    evaluations = [0.0] * len(moves)
    for i, move in enumerate(moves):
        game_state_ = game_state.copy()
        game_state_.make_move(move)
        eval, _ = model.forward(model.state2tensor(game_state_))
        evaluations[i] = eval.item()
    max_value = max(evaluations)
    max_index = evaluations.index(max_value)
    print([round(eval, 2) for eval in evaluations])

    return moves[max_index]


def best_prior_find_move(game_state: TicTacToeGameState, model: AlphaZero):
    moves = game_state.available_moves()
    torch_board = model.state2tensor(game_state)
    _, priors = model.forward(torch_board)

    mask = TicTacToeGameState.get_masks(torch_board)
    priors = priors * mask
    priors = priors.reshape((1, 9))
    priors = torch.nn.functional.normalize(priors, p=1, dim=1)

    new_prior = []

    print(priors.view(9))
    print(priors.view(9)[0].item())

    for p in priors.view(9):
        if p.item() != 0.0:
            new_prior.append(p.item())

    print([round(prior, 2) for prior in new_prior])

    max_value = max(new_prior)
    max_index = new_prior.index(max_value)
    return moves[max_index]


def print_data(game_state: TicTacToeGameState, model: AlphaZero) -> None:
    torch_board = model.state2tensor(game_state)
    print(f"Evaluation after move: {model(torch_board)[0][0][0].item():.4f}")
    print(" ")


if __name__ == '__main__':
    main()
