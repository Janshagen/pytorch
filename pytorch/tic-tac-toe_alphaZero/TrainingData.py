from datetime import datetime

import torch
from TicTacToeModel import AlphaZero, Loss
from Visualizer import Visualizer


class TrainingData:
    def __init__(self, model: AlphaZero,
                 loss: Loss,
                 optimizer: torch.optim.Optimizer,
                 N_BATCHES: int):

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.visualizer = Visualizer()

        self.device: torch.device = model.device
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[N_BATCHES//10, 8*N_BATCHES//10], gamma=0.1
        )

    def save_model(self):
        path = self.get_save_path()
        with open(path, "w+") as file:
            torch.save(self.model.state_dict(), file.name)

    @staticmethod
    def get_save_path() -> str:
        current_time = datetime.today().strftime("%Y-%m-%d %H:%M")
        game_dir = '/home/anton/skola/egen/pytorch/tic-tac-toe_alphaZero'
        return game_dir + f'/models/AlphaZero{current_time}.pth'
