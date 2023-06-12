import random

import numpy as np
import torch
import torch.nn as nn
from MCTS import MCTS
from DeepLearningData import DeepLearningData
from GameRules import TicTacToeGameState
from TicTacToeModel import AlphaZero, Loss

# Constants
LOAD_MODEL = False
SAVE_MODEL = True
FILE = '/home/anton/skola/egen/pytorch/tic-tac-toe/alpha_zero.pth'

LEARNING_RATE = 0.01
N_EPOCHS = 100_000

SIMULATIONS = 30
UCB1 = 1.4


def main() -> None:
    mcts = MCTS(UCB1, sim_number=SIMULATIONS)
    learning_data = create_learning_data()

    train(mcts, learning_data)
    validate(learning_data)

    if SAVE_MODEL:
        learning_data.save_model()


def create_learning_data():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AlphaZero(device)
    if LOAD_MODEL:
        model = DeepLearningData.load_model(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    loss = Loss()

    learning_data = DeepLearningData(model, device, loss, optimizer)
    return learning_data


def train(mcts: MCTS, learning_data: DeepLearningData) -> None:
    assert learning_data.loss
    assert learning_data.optimizer
    for epoch in range(N_EPOCHS):
        board_states, result, visits = game(mcts, learning_data)

        num_moves = board_states.shape[0]
        result = result.expand((num_moves, 1))

        evaluations, policies = learning_data.model(board_states)

        error = learning_data.loss(evaluations, result, policies, visits)
        error.backward()

        learning_data.optimizer.step()
        learning_data.optimizer.zero_grad()

        if (epoch + 1) % (N_EPOCHS/10) == 0:
            print_info(epoch, evaluations, error)

            if SAVE_MODEL:
                learning_data.save_model()


def print_info(epoch, evaluations, error):
    print(
        f'Epoch [{epoch+1}/{N_EPOCHS}], Loss: {error.item():.8f}',
        f'evaluation: {evaluations[0].item():.4f}')


def game(mcts: MCTS, learning_data: DeepLearningData) -> \
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    game_state = create_game_state()
    board_states = torch.zeros((4, 3, 3, 3), device=learning_data.device)
    board_states[:][2] = torch.ones((3, 3)) * game_state.player

    all_visits = torch.tensor([], device=learning_data.device)
    while True:
        game_state = perform_game_action(mcts, game_state, learning_data)
        all_visits = add_number_of_visits(mcts, learning_data, all_visits)
        board_states = add_new_rotated_board_states(
            game_state,
            learning_data,
            board_states
        )

        status = game_state.game_status()
        if TicTacToeGameState.game_over(status):
            visits = torch.tensor([0.111]*9, dtype=torch.float32,
                                  device=learning_data.device)
            for _ in range(4):
                all_visits = torch.cat((all_visits, visits.expand((1, -1))), dim=0)

            return board_states, torch.tensor(
                [status],
                dtype=torch.float32, device=learning_data.device), all_visits


def create_game_state():
    player = random.choice([1, -1])
    board = np.zeros((3, 3), dtype=np.int8)
    return TicTacToeGameState(board, player)


def perform_game_action(mcts: MCTS, game_state: TicTacToeGameState,
                        learning_data: DeepLearningData) -> TicTacToeGameState:
    move = mcts.find_move(game_state, learning_data)
    game_state.make_move(move)
    return game_state


def add_number_of_visits(mcts: MCTS, learning_data: DeepLearningData,
                         all_visits: torch.Tensor) -> torch.Tensor:
    visits = [0]*9
    for child in mcts.root.children:
        visits[TicTacToeGameState.move2index(child.move)] = child.visits

    visits = torch.tensor(visits, dtype=torch.float32, device=learning_data.device)
    visits = nn.functional.normalize(visits, dim=0, p=1)
    for _ in range(4):
        all_visits = torch.cat((all_visits, visits.expand((1, -1))), dim=0)
    return all_visits


def add_new_rotated_board_states(game_state: TicTacToeGameState,
                                 learning_data: DeepLearningData,
                                 board_states: torch.Tensor) -> torch.Tensor:
    torch_board = learning_data.model.state2tensor(game_state)
    for i in range(4):
        permutation = torch_board.rot90(k=i, dims=(2, 3))
        board_states = torch.cat((board_states, permutation), dim=0)
    return board_states


def validate(learning_data: DeepLearningData) -> None:
    win1 = np.array([[1, 1, 1],
                     [-1, 0, -1],
                     [-1, 0, 0]])
    game_state1 = TicTacToeGameState(win1, -1)
    torch_board1 = learning_data.model.state2tensor(game_state1)

    draw = np.array([[1, -1, 1],
                     [-1, 1, -1],
                     [-1, 1, -1]])
    game_state2 = TicTacToeGameState(draw, 1)
    torch_board2 = learning_data.model.state2tensor(game_state2)

    win2 = np.array([[-1, 1, 1],
                     [-1, 1, -1],
                     [-1, -1, 1]])
    game_state3 = TicTacToeGameState(win2, 1)
    torch_board3 = learning_data.model.state2tensor(game_state3)

    with torch.no_grad():
        print('win 1:', learning_data.model(torch_board1))
        print('draw:', learning_data.model(torch_board2))
        print('win 2:', learning_data.model(torch_board3))


if __name__ == '__main__':
    main()
