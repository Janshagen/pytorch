import torch

OLD_GAMES = '/home/anton/skola/egen/pytorch/connect4_alphaZero/games_5dim/'
NEW_GAMES = '/home/anton/skola/egen/pytorch/connect4_alphaZero/games_current_player_perspective/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_printoptions(threshold=100000)

# # to 5 dim games
# for file_number in range(10):
#     data = [torch.tensor([], device=device) for _ in range(4)]
#     with open(OLD_GAMES + f"game_{file_number}.pt", 'rb') as file:
#         old_data = torch.load(file)
#         boards = old_data[0]
#         assert isinstance(boards, torch.Tensor)

#         new_board = torch.ones((boards.shape[0], 1, 6, 7),
#                                dtype=torch.float32, device=device)
#         ones = torch.ones((1, 6, 7),
#                           dtype=torch.float32, device=device)

#         for i, board in enumerate(boards):
#             new_board[i, board[0] == ones] = 0
#             new_board[i, board[1] == ones] = 0

#         data[0] = torch.cat((new_board, boards), dim=1)

#         for j in range(1, 4):
#             old_data[j] = old_data[j].to(device)
#             data[j] = torch.cat((data[j], old_data[j]))

#     with open(NEW_GAMES + f"game_{file_number}.pt", 'wb') as file:
#         torch.save(data, file)


# to current player perspective
ones = torch.ones((1, 6, 7),
                  dtype=torch.float32, device=device)
for file_number in range(1):
    data = [torch.tensor([], device=device) for _ in range(4)]
    with open(OLD_GAMES + f"game_{file_number}.pt", 'rb') as file:
        old_data = torch.load(file)

        # boards
        boards = old_data[0]
        game_lengths = old_data[3]
        assert isinstance(boards, torch.Tensor)

        new_boards = torch.ones((boards.shape[0], 3, 6, 7),
                                dtype=torch.float32, device=device)

        i = 0
        for game in game_lengths:
            game = int(game.item())
            first = boards[i, 1]
            second = boards[i, 2]

            if first.any():
                # "first" player starts game
                new_boards[i, 0] = boards[i, 1]
                new_boards[i, 1] = boards[i, 2]
                for move in range(game):
                    if (i % 4 == 0) or (i % 4 == 1):
                        new_boards[i, 0] = boards[i, 1]
                        new_boards[i, 1] = boards[i, 2]
                    else:
                        new_boards[i, 0] = boards[i, 2]
                        new_boards[i, 1] = boards[i, 1]

                    empty = boards[i, 0]
                    new_boards[i, 2] = empty
                    i += 1

            elif second.any():
                new_boards[i, 0] = boards[i, 2]
                new_boards[i, 1] = boards[i, 1]
                for move in range(game):
                    if (i % 4 == 2) or (i % 4 == 3):
                        new_boards[i, 0] = boards[i, 2]
                        new_boards[i, 1] = boards[i, 1]
                    else:
                        new_boards[i, 0] = boards[i, 1]
                        new_boards[i, 1] = boards[i, 2]

                    empty = boards[i, 0]
                    new_boards[i, 2] = empty
                    i += 1
            else:
                print("fuck")

        data[0] = new_boards

        exit()

        # results
        results = old_data[1]

        new_result = torch.zeros(results.shape)

        old_res = 10
        a = 10
        for i, res in enumerate(torch.flip(results, dims=(0,))):
            if i % 2 != 0:
                continue

            if res != old_res:
                old_res = res
                a = 1

            new_result[i] = a
            new_result[i+1] = a
            a = -a
        new_result = torch.flip(new_result, dims=(0,))

        data[1] = torch.cat((data[1], new_result))

        # other
        for j in range(2, 4):
            data[j] = torch.cat((data[j], old_data[j]))

    with open(NEW_GAMES + f"game_{file_number}.pt", 'wb') as file:
        torch.save(data, file)
