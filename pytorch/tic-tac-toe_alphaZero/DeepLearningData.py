from typing import Optional

import torch
from TicTacToeModel import AlphaZero, Loss

FILE = '/home/anton/skola/egen/pytorch/tic-tac-toe/alpha_zero.pth'


class DeepLearningData:
    def __init__(self, model: AlphaZero,
                 device: torch.device,
                 loss: Optional[Loss] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None):
        self.model = model
        self.device = device

        self.loss = loss
        self.optimizer = optimizer

    def save_model(self):
        torch.save(self.model.state_dict(), FILE)

    @staticmethod
    def load_model(device: torch.device):
        model = AlphaZero(device)
        model.load_state_dict(torch.load(FILE))
        model.to(device)
        model.eval()
        return model
