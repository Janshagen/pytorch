import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([[1, 2, 3],
                  [2, 4, 2]])
c = torch.tensor([[4, 5, 1],
                  [2, 7, 2],
                  [2, 4, 2]])

# with open('/home/anton/skola/egen/pytorch/tic-tac-toe_alphaZero/games.pt', "wb") as file:
#     torch.save([a, b, c], file)

with open('/home/anton/skola/egen/pytorch/tic-tac-toe_alphaZero/games.pt', "rb") as file:
    hej = torch.load(file)

visits = hej[2]
print(visits[0])
print(visits[1])
print(visits[2])
print(visits[3])
