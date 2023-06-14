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

LEARNING_RATE = 0.2
N_BATCHES = 20
BATCH_SIZE = 10

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
    model = create_model(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    loss = Loss()

    learning_data = DeepLearningData(model, device, loss, optimizer, N_BATCHES)
    return learning_data


def create_model(device):
    model = AlphaZero(device)
    if LOAD_MODEL:
        model = DeepLearningData.load_model(device)
    return model


def train(mcts: MCTS, learning_data: DeepLearningData) -> None:
    assert learning_data.loss
    assert learning_data.optimizer
    for batch in range(N_BATCHES):
        boards = torch.tensor([], device=learning_data.device)
        results = torch.tensor([], device=learning_data.device)
        visits = torch.tensor([], device=learning_data.device)
        for sample in range(BATCH_SIZE):
            game_boards, game_result, game_visits = get_game_data(mcts, learning_data)

            boards = torch.cat((boards, game_boards), dim=0)
            results = torch.cat((results, game_result), dim=0)
            visits = torch.cat((visits, game_visits), dim=0)

        evaluations, policies = learning_data.model(boards)

        error = learning_data.loss(evaluations, results, policies, visits)
        error.backward()

        learning_data.optimizer.step()
        learning_data.optimizer.zero_grad()
        learning_data.scheduler.step()

        if (batch + 1) % (N_BATCHES/10) == 0:
            print_info(batch, evaluations, results, error)
            if SAVE_MODEL:
                learning_data.save_model()


def get_game_data(mcts: MCTS, learning_data: DeepLearningData) -> \
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    game_board_states, game_result, game_visits = game(mcts, learning_data)

    num_moves = game_board_states.shape[0]
    game_result = game_result.expand((num_moves, 1))
    return game_board_states, game_result, game_visits


def print_info(batch: int, evaluations: torch.Tensor,
               result: torch.Tensor, error: torch.Tensor) -> None:
    print(
        f'Batch [{batch+1}/{N_BATCHES}], Loss: {error.item():.8f}',
        f'evaluation: {evaluations[0].item():.4f}, result: {result[0][0].item()}')


def game(mcts: MCTS, learning_data: DeepLearningData) -> \
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    game_state = TicTacToeGameState.new_game(random.choice([1, -1]))
    boards = torch.zeros((4, 3, 3, 3), device=learning_data.device)
    if game_state.player == 1:
        for i in range(4):
            boards[i][-1] = torch.ones((3, 3))

    all_visits = torch.tensor([], device=learning_data.device)
    while True:
        game_state = perform_game_action(mcts, learning_data, game_state)
        all_visits = add_number_of_visits(mcts, learning_data, all_visits)
        boards = add_new_rotated_board_states(learning_data, game_state, boards)

        status = game_state.game_status()
        if TicTacToeGameState.game_over(status):
            visits = torch.tensor([0.111]*9, dtype=torch.float32,
                                  device=learning_data.device)
            for _ in range(4):
                all_visits = torch.cat((all_visits, visits.expand((1, -1))), dim=0)

            result = torch.tensor([status], dtype=torch.float32,
                                  device=learning_data.device)
            return boards, result, all_visits


def perform_game_action(mcts: MCTS, learning_data: DeepLearningData,
                        game_state: TicTacToeGameState) -> TicTacToeGameState:
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


def add_new_rotated_board_states(learning_data: DeepLearningData,
                                 game_state: TicTacToeGameState,
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
