# import numpy as np
import torch
from Connect4Model import AlphaZero
# from GameRules import Connect4GameState

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def add_new_flipped_boards():
#     all_boards = torch.tensor([], device=device)

#     model = AlphaZero()
#     state = Connect4GameState.new_game()
#     state.board = np.array([[0,  1,  1, -1, 0,  1,  1],
#                             [0, -1, -1, -1,  0, -1, -1],
#                             [1,  1,  1, -1,  0,  1,  1],
#                             [-1, -1,  1,  1,  0,  1, -1],
#                             [1,  1, -1, -1,  1, -1, 1],
#                             [-1, -1,  1, -1, -1,  1, -1]], dtype=np.int8)

#     torch_board = model.state2tensor(state)
#     print(torch_board)
#     all_boards = add_flipped_states(all_boards, torch_board, flip_dims=(3,))
#     print(all_boards)
#     return all_boards


# def add_new_flipped_visits() -> torch.Tensor:
#     all_visits = torch.tensor([], device=device)

#     visits = torch.arange(7, dtype=torch.float32, device=device).reshape((1, 7))

#     print(visits)
#     all_visits = add_flipped_states(all_visits, visits, flip_dims=(1,))
#     print(all_visits)
#     return all_visits


# def add_flipped_states(
#         stack: torch.Tensor,
#         new_state: torch.Tensor,
#         flip_dims: tuple) -> torch.Tensor:
#     stack = torch.cat((stack, new_state), dim=0)
#     new_state = torch.flip(new_state, dims=flip_dims)
#     stack = torch.cat((stack, new_state), dim=0)
#     return stack


a = torch.arange(30).reshape((-1, 1, 3))
print(a)

step = 5
b = a.shape[0]//step

torch.manual_seed(42)
r = torch.randperm(a.shape[0])
a = a[r]

print(r)


# scheduler = torch.optim.lr_scheduler.MultiStepLR(
#     torch.optim.SGD(AlphaZero().parameters(),
#                     lr=1,
#                     weight_decay=0.1), milestones=[10, 15], gamma=0.1
# )

# for i in range(20):
#     print(scheduler.get_last_lr())
#     scheduler.step()
