import matplotlib.pyplot as plt
import numpy as np
import torchvision
from TicTacToeModel import AlphaZero
from torch.utils.tensorboard.writer import SummaryWriter
from GameRules import TicTacToeGameState


class Visualizer:

    def __init__(self) -> None:
        self.writer = SummaryWriter('tic-tac-toe_alphaZero/runs/')

    def visualize_model(self, model: AlphaZero) -> None:
        win1_numpy = np.array([[1, 1, 1],
                               [-1, 0, -1],
                               [-1, 0, 0]]
                              )
        win1_gamestate = TicTacToeGameState(win1_numpy, -1, 1)
        torch_board = model.state2tensor(win1_gamestate)
        grid = torchvision.utils.make_grid(torch_board)

        mask = TicTacToeGameState.get_masks(torch_board)
        grid_model = grid * mask

        self.writer.add_graph(model, grid_model)

    def add_loss(self, loss: float, step: int) -> None:
        self.writer.add_scalar("Training Loss", loss, step)

    def close(self) -> None:
        self.writer.close()

    def matplotlib_imshow(self, img, one_channel=False):
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
    visualizer = Visualizer()
