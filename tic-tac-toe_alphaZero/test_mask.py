import torch

board = torch.arange(27*2, dtype=torch.float32).reshape(2, 3, 3, 3)
p = torch.arange(18, dtype=torch.float32).reshape(2, 1, 3, 3) + 2

board[0, 0, 1, 2] = 40

ones = torch.ones(p.shape[1:])*40

mask = torch.ones(p.shape)

print(board)
print(p)
illeagal_indecies = ((board[:, 0] == ones) + (board[:, 1] == ones)).unsqueeze(dim=1)
mask[illeagal_indecies] = 0
print(p*mask)
