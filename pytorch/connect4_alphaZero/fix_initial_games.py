import torch

OLD_GAMES = '/home/anton/skola/egen/pytorch/connect4_alphaZero/games/'
NEW_GAMES = '/home/anton/skola/egen/pytorch/connect4_alphaZero/games_5dim/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for file_number in range(10):
    data = [torch.tensor([], device=device) for _ in range(4)]
    with open(OLD_GAMES + f"game_{file_number}.pt", 'rb') as file:
        new_data = torch.load(file)
        boards = new_data[0]
        assert isinstance(boards, torch.Tensor)

        first_layer = torch.ones((boards.shape[0], 1, 6, 7),
                                 dtype=torch.float32, device=device)
        ones = torch.ones((1, 6, 7),
                          dtype=torch.float32, device=device)

        for i, board in enumerate(boards):
            first_layer[i, board[0] == ones] = 0
            first_layer[i, board[1] == ones] = 0

        data[0] = torch.cat((first_layer, boards), dim=1)

        for j in range(1, 4):
            new_data[j] = new_data[j].to(device)
            data[j] = torch.cat((data[j], new_data[j]))

    with open(NEW_GAMES + f"game_{file_number}.pt", 'wb') as file:
        torch.save(data, file)
