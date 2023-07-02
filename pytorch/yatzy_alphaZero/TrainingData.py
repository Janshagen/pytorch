from datetime import datetime

import torch
from YatzyModel import AlphaZero, Loss


class TrainingData:
    def __init__(self, model: AlphaZero,
                 loss: Loss,
                 optimizer: torch.optim.Optimizer,
                 N_BATCHES: int):

        self.model = model
        self.loss = loss
        self.optimizer = optimizer

        self.device: torch.device = model.device
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[2*N_BATCHES//10, 8*N_BATCHES//10], gamma=0.1
        )

    def save_model(self):
        path = self.get_save_path()
        with open(path, "w+") as file:
            torch.save(self.model.state_dict(), file.name)

    @staticmethod
    def get_save_path() -> str:
        current_time = datetime.today().strftime("%Y-%m-%d %H:%M")
        return AlphaZero.MODEL_PATH + f'AlphaZero{current_time}.pth'
