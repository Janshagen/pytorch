import random

import numpy as np
import torch
import torch.nn as nn
from AI import MCTS_find_move
from gameplay import (game_over, game_status, make_move,
                      next_player, move2index)
from MCTSData import MCTSData
from TicTacToeModel import AlphaZero, Loss

# Constants
LOAD_MODEL = False
SAVE_MODEL = False
FILE = '/home/anton/skola/egen/pytorch/tic-tac-toe/alpha_zero.pth'

LEARNING_RATE = 0.01
N_EPOCHS = 100_000

OUTPUT_SIZE = 3
HIDDEN_SIZE1 = 72
HIDDEN_SIZE2 = 72

SIMULATIONS = 30
UCB1 = 1.4


def train(data: MCTSData, optimizer: torch.optim.Optimizer, loss: Loss) -> None:
    for epoch in range(N_EPOCHS):
        game_states, result, visits = game(data)
        evaluations, policies = data.model(game_states)

        player = random.choice([1, -1])
        board = np.zeros((3, 3), dtype=np.int8)
        data.board = board
        data.player = player

        num_moves = game_states.shape[0]
        result = result.expand((num_moves, 1))

        error = loss(evaluations, result, policies, visits)

        error.backward()

        optimizer.step()
        optimizer.zero_grad()

        if (epoch + 1) % 100 == 0:
            print(
                f'Epoch [{epoch+1}/{N_EPOCHS}], Loss: {error.item():.8f}',
                f'evaluation: {evaluations[0]:.4f}, result: {result.item()}')


def game(data: MCTSData) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    board_states = torch.zeros((4, 3, 3, 3), device=data.device)
    board_states[:][2] = torch.ones((3, 3)) * data.player

    all_visits = torch.tensor([], device=data.device)
    while True:
        perform_game_action(data)
        all_visits = add_number_of_visits(data, all_visits)
        board_states = add_new_rotated_board_states(data, board_states)

        status = game_status(data.board)
        if game_over(status):
            visits = torch.tensor([0.111]*9, dtype=torch.float32, device=data.device)
            for _ in range(4):
                all_visits = torch.cat((all_visits, visits.expand((1, -1))), dim=0)

            return board_states, torch.tensor(
                [status],
                dtype=torch.float32, device=data.device), all_visits


def perform_game_action(data: MCTSData) -> None:
    move = MCTS_find_move(data)
    make_move(data.board, data.player, move)
    data.player = next_player(data.player)


def add_number_of_visits(data: MCTSData, all_visits: torch.Tensor) -> torch.Tensor:
    visits = [0]*9
    for child in data.root.children:
        visits[move2index(child.move)] = child.visits

    visits = torch.tensor(visits, dtype=torch.float32, device=data.device)
    visits = nn.functional.normalize(visits, dim=0)
    for _ in range(4):
        all_visits = torch.cat((all_visits, visits.expand((1, -1))), dim=0)
    return all_visits


def add_new_rotated_board_states(data: MCTSData, board_states: torch.Tensor) \
        -> torch.Tensor:
    for i in range(4):
        permutation = data.model.board2tensor(
            data.board, data.player).rot90(k=i, dims=(2, 3))
        board_states = torch.cat((board_states, permutation), dim=0)
    return board_states


def validate(data: MCTSData) -> None:
    win1 = np.array([[1, 1, 1],
                     [-1, 0, -1],
                     [-1, 0, 0]])

    draw = np.array([[1, -1, 1],
                     [-1, 1, -1],
                     [-1, 1, -1]])

    win2 = np.array([[-1, 1, 1],
                     [-1, 1, -1],
                     [-1, -1, 1]])

    with torch.no_grad():
        print('win 1:', torch.softmax(
            data.model(win1, -1, data.device), 2))
        print('draw:', torch.softmax(
            data.model(draw, 1, data.device), 2))
        print('win 2:', torch.softmax(
            data.model(win2, 1, data.device), 2))


def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(device)

    player = random.choice([1, -1])
    board = np.zeros((3, 3), dtype=np.int8)
    data = MCTSData(board, player, UCB1, model, device, sim_number=SIMULATIONS)

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    loss = Loss()

    train(data, optimizer, loss)
    validate(data)

    if SAVE_MODEL:
        torch.save(model.state_dict(), FILE)


def load_model(device: torch.device):
    model = AlphaZero(device)
    if LOAD_MODEL:
        model.load_state_dict(torch.load(FILE))
        model.to(device)
        model.eval()
    return model


if __name__ == '__main__':
    main()
