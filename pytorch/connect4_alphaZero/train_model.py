import random

import numpy as np
import torch
import torch.nn as nn
from TrainingData import TrainingData
from GameRules import Connect4GameState
from MCTS import MCTS
from Connect4Model import AlphaZero, Loss

# Constants
LOAD_MODEL = False
SAVE_MODEL = True

LEARNING_RATE = 0.2
MOMENTUM = 0.9
WEIGHT_DECAY = 0.01

N_BATCHES = 3_000
BATCH_SIZE = 5

SIMULATIONS = 50
EXPLORATION_RATE = 5


def main() -> None:
    learning_data = create_learning_data()
    mcts = MCTS(learning_data.model, EXPLORATION_RATE, sim_number=SIMULATIONS)

    print("Training Started")
    train(mcts, learning_data)
    validate(learning_data)

    if SAVE_MODEL:
        learning_data.save_model()


def create_learning_data() -> TrainingData:
    model = create_model()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=LEARNING_RATE,
                                momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY)
    loss = Loss()

    learning_data = TrainingData(model, loss, optimizer, N_BATCHES)
    return learning_data


def create_model() -> AlphaZero:
    model = AlphaZero()
    if LOAD_MODEL:
        model = model.load_model()
    model.train()
    return model


def train(mcts: MCTS, learning_data: TrainingData) -> None:
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


def get_game_data(mcts: MCTS, learning_data: TrainingData) -> \
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    game_board_states, game_result, game_visits = game(mcts, learning_data)

    num_moves = game_board_states.shape[0]
    game_result = game_result.expand((num_moves, 1))
    return game_board_states, game_result, game_visits


def print_info(batch: int, evaluations: torch.Tensor,
               result: torch.Tensor, error: torch.Tensor) -> None:
    print(
        f'Batch [{batch+1}/{N_BATCHES}], Loss: {error.item():.8f},',
        f'evaluation: {evaluations[-1].item():.4f}, result: {result[0][0].item()}')


def game(mcts: MCTS, learning_data: TrainingData) -> \
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    game_state = Connect4GameState.new_game(random.choice([1, -1]))
    boards = torch.zeros((2, 3, 6, 7), device=learning_data.device)
    if game_state.player == 1:
        for i in range(2):
            boards[i][-1] = torch.ones((6, 7))

    all_visits = torch.tensor([], device=learning_data.device)
    while True:
        game_state = perform_game_action(mcts, game_state)
        all_visits = add_number_of_visits(mcts, learning_data, all_visits)
        boards = add_new_flipped_board_states(learning_data, game_state, boards)

        if game_state.game_over():
            visits = torch.tensor([0.142857143]*7, dtype=torch.float32,
                                  device=learning_data.device)
            for _ in range(2):
                all_visits = torch.cat((all_visits, visits.unsqueeze(dim=0)), dim=0)

            result = torch.tensor([game_state.get_status()], dtype=torch.float32,
                                  device=learning_data.device)
            return boards, result, all_visits


def perform_game_action(mcts: MCTS,
                        game_state: Connect4GameState) -> Connect4GameState:
    move = mcts.find_move(game_state)
    game_state.make_move(move)
    return game_state


def add_number_of_visits(mcts: MCTS, learning_data: TrainingData,
                         all_visits: torch.Tensor) -> torch.Tensor:
    visits = [0]*7
    for child in mcts.root.children:
        visits[child.move] = child.visits

    visits = torch.tensor(visits, dtype=torch.float32, device=learning_data.device)
    visits = nn.functional.normalize(visits, dim=0, p=1)
    for _ in range(2):
        all_visits = torch.cat((all_visits, visits.unsqueeze(dim=0)), dim=0)
    return all_visits


def add_new_flipped_board_states(learning_data: TrainingData,
                                 game_state: Connect4GameState,
                                 board_states: torch.Tensor) -> torch.Tensor:
    torch_board = learning_data.model.state2tensor(game_state)
    board_states = torch.cat((board_states, torch_board), dim=0)
    torch_board = torch.flip(torch_board, dims=(3,))
    board_states = torch.cat((board_states, torch_board), dim=0)
    return board_states


def validate(learning_data: TrainingData) -> None:
    win1 = np.array([[0,  1,  1, -1, 0,  1,  1],
                     [0, -1, -1, -1,  0, -1, -1],
                     [1,  1,  1, -1,  0,  1,  1],
                     [-1, -1,  1,  1,  0,  1, -1],
                     [1,  1, -1, -1,  1, -1, 1],
                     [-1, -1,  1, -1, -1,  1, -1]])
    game_state1 = Connect4GameState(win1, -1)
    torch_board1 = learning_data.model.state2tensor(game_state1)

    draw = np.array([[1,  1, -1,  1, -1, -1, 1],
                     [-1,  -1,  1,  1, -1,  1, -1],
                     [1, -1,  1, -1, -1, -1,  1],
                     [-1,  1,  1, -1,  1, -1,  1],
                     [-1, -1, -1,  1, -1,  1, -1],
                     [1,  1,  1, -1,  1, -1, 1]])
    game_state2 = Connect4GameState(draw, -1)
    torch_board2 = learning_data.model.state2tensor(game_state2)

    win2 = np.array([[0,  0,  0,  0,  0,  0,  0],
                    [0, -1, -1, -1, -1,  1,  0],
                    [-1,  1,  1,  1, -1, 1, 0],
                    [1, -1, -1,  1,  1, -1, 0],
                    [-1,  1,  1, -1, 1, -1,  1],
                    [1, -1,  1, -1, -1, -1,  1]])
    game_state3 = Connect4GameState(win2, 1)
    torch_board3 = learning_data.model.state2tensor(game_state3)

    with torch.no_grad():
        print('win 1   :', learning_data.model(torch_board1))
        print('draw (0):', learning_data.model(torch_board2))
        print('win -1  :', learning_data.model(torch_board3))


if __name__ == '__main__':
    main()
