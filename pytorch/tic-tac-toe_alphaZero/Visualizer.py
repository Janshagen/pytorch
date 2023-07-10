import matplotlib.pyplot as plt
import numpy as np
import torchvision
from TicTacToeModel import AlphaZero
from torch.utils.tensorboard.writer import SummaryWriter
from GameRules import TicTacToeGameState


def main():
    model = AlphaZero()
    model.train()

    writer = SummaryWriter('tic-tac-toe_alphaZero/runs/')

    win1_numpy = np.array([[1, 1, 1],
                           [-1, 0, -1],
                           [-1, 0, 0]]
                          )
    win1 = TicTacToeGameState(win1_numpy, -1, 1)
    torch_board = model.state2tensor(win1)
    grid = torchvision.utils.make_grid(torch_board)

    # matplotlib_imshow(torch_board, one_channel=True)
    # writer.add_image("win1", grid)

    mask = TicTacToeGameState.get_masks(torch_board)
    grid = grid * mask

    writer.add_graph(model, grid)
    writer.close()


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img.to("cpu")
    img = img / 2 + 0.5     # unnormalize
    np_img = img.numpy()
    if one_channel:
        plt.imshow(np_img, cmap="Greys")
    else:
        plt.imshow(np.transpose(np_img, (1, 2, 0)))


if __name__ == '__main__':
    main()
