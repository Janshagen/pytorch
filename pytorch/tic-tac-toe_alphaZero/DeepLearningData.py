import os
import re
import time
from typing import Optional

import torch
from TicTacToeModel import AlphaZero, Loss


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
        path = self.get_save_file()
        with open(path, "w+") as file:
            torch.save(self.model.state_dict(), file.name)

    @staticmethod
    def load_model(device: torch.device, file: Optional[str] = None):
        model = AlphaZero(device)
        model.load_state_dict(torch.load(DeepLearningData.get_load_file(file)))
        model.to(device)
        model.eval()
        return model

    @staticmethod
    def get_save_file() -> str:
        game_dir = '/home/anton/skola/egen/pytorch/tic-tac-toe_alphaZero'
        return game_dir + f'/models/AlphaZero{int(time.time())}.pth'

    @staticmethod
    def get_load_file(file: Optional[str] = None) -> str:
        model_path = '/home/anton/skola/egen/pytorch/tic-tac-toe_alphaZero/models/'

        model_name = re.compile(".*?AlphaZero(.*?).pth")
        files = os.listdir(model_path)

        times = []
        for file in files:
            matches = model_name.match(file)
            if not matches:
                times.append(0)
                continue
            times.append(int(matches.group(1)))
        max_time = max(times)
        max_index = times.index(max_time)

        return model_path + files[max_index]
