from typing import Optional
import os
import re

import torch
from TicTacToeModel import AlphaZero, Loss
import time


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
        torch.save(self.model.state_dict(), self.get_save_file())

    @staticmethod
    def load_model(device: torch.device, file: Optional[str] = None):
        model = AlphaZero(device)
        model.load_state_dict(torch.load(DeepLearningData.get_load_file(file)))
        model.to(device)
        model.eval()
        return model

    @staticmethod
    def get_save_file() -> str:
        return f'/home/anton/skola/egen/pytorch/tic-tac-toe_alphaZero \
                /models/AlphaZero{int(time.time())}.pth'

    @staticmethod
    def get_load_file(file: Optional[str] = None) -> str:
        regex = re.compile(".*?AlphaZero(.*?).pth")
        files = os.listdir(
            '/home/anton/skola/egen/pytorch/tic-tac-toe_alphaZero/models/'
        )

        times = []
        for file in files:
            matches = regex.match(file)
            if not matches:
                times.append(0)
                continue
            times.append(int(matches.group(1)))
        max_time = max(times)
        max_index = times.index(max_time)

        return files[max_index]


if __name__ == '__main__':
    DeepLearningData.get_load_file()
