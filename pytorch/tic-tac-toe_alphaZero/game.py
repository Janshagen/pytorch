import pygame
import torch
from MCTS import MCTS
from GameRules import TicTacToeGameState
from DeepLearningData import DeepLearningData
from TicTacToeModel import AlphaZero
from interface import (chooseConfig, draw, gameOver, initializeGame,
                       resolveEvent)


# Configurations
SIMULATIONS = 100
WIDTH = 200
UCB1 = 1.4


def main() -> None:
    sims = chooseConfig(SIMULATIONS)
    screen, frame = initializeGame(WIDTH)

    game_state = TicTacToeGameState.new_game()
    learning_data = create_learning_data()
    mcts = MCTS(UCB1, sim_number=sims)

    draw(screen, frame, game_state, WIDTH)

    result = game(mcts, game_state, learning_data,  screen, frame)
    if not gameOver(screen, result, WIDTH):
        main()


def create_learning_data():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AlphaZero(device)
    model = DeepLearningData.load_model(device)

    learning_data = DeepLearningData(model, device)
    return learning_data


def game(mcts: MCTS, game_state: TicTacToeGameState, learning_data: DeepLearningData,
         screen: pygame.surface.Surface, frame: pygame.Surface) -> int:
    while True:
        # Human
        if game_state.player == -1:
            move = resolveEvent(game_state, WIDTH)
            if move:
                game_state.make_move(move)

        # AI
        elif game_state.player == 1:
            move = mcts.find_move(game_state, learning_data)
            game_state.make_move(move)
            resolveEvent(game_state, WIDTH)

            torch_board = learning_data.model.state2tensor(game_state)
            print(f"evaluation: {learning_data.model(torch_board)[0][0][0].item():.4f}")
            print(f"policy: {learning_data.model(torch_board)[1]}")

        draw(screen, frame, game_state, WIDTH)

        status = game_state.game_status()
        if TicTacToeGameState.game_over(status):
            return status


if __name__ == '__main__':
    main()
