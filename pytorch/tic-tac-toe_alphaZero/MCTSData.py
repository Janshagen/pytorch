import numpy as np
import numpy.typing as npt
import torch
from Node import Node
from TicTacToeModel import ConvModel, LinearModel

board_type = npt.NDArray[np.int8]


class MCTSData:
    def __init__(self, board: board_type, player: int, UCB1: float,
                 model: LinearModel | ConvModel, device: torch.device,
                 sim_time: float = np.inf, sim_number: int = 1_000_000,
                 cutoff: int = 0) -> None:

        self.board = board
        self.player = player
        self.UCB1 = UCB1
        self.sim_time = sim_time
        self.sim_number = sim_number
        self.cutoff = cutoff

        self.model = model
        self.device = device

        self.root: Node
