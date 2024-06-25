import matplotlib.pyplot as plt
import numpy as np
import torchvision
from Connect4Model import AlphaZero
from torch.utils.tensorboard.writer import SummaryWriter
from GameRules import Connect4GameState


class Visualizer:

    def __init__(self) -> None:
        self.writer = SummaryWriter('connect4_alphaZero/runs/')

    def visualize_model(self, model: AlphaZero) -> None:
        win1_numpy = np.array([[0,  1,  1, -1, 0,  1,  1],
                               [0, -1, -1, -1,  0, -1, -1],
                               [1,  1,  1, -1,  0,  1,  1],
                               [-1, -1,  1,  1,  0,  1, -1],
                               [1,  1, -1, -1,  1, -1, 1],
                               [-1, -1,  1, -1, -1,  1, -1]])
        win1_gamestate = Connect4GameState(win1_numpy, -1, 1)
        torch_board = model.state2tensor(win1_gamestate)
        grid = torchvision.utils.make_grid(torch_board)

        mask = Connect4GameState.get_masks(torch_board)
        grid_model = grid * mask

        self.writer.add_graph(model, grid_model)

    def add_loss(self, loss_type: str, loss: float, step: int) -> None:
        self.writer.add_scalar(loss_type, loss, step)

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
