import numpy as np
import pygame
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from train_model import Model
import matplotlib.pyplot as plt

RES = 20
WIDTH = 28*RES
BAR_WIDTH = 200

WHITE = (255, 255, 255)
LIGHT_WHITE = (200, 200, 200)
GREY = (180, 180, 180)
DARK_GREY = (100, 100, 100)
BLACK = (0, 0, 0)

FILE = '/home/anton/skola/egen/pytorch/mnist/model.pth'
INPUT_SIZE = 28*28
OUTPUT_SIZE = 10
HIDDEN_SIZE1 = 512
HIDDEN_SIZE2 = 256


class Button:
    w = 65

    def __init__(self, y, display, message) -> None:
        self.y = y
        self.message = message
        self.font = pygame.font.Font(None, 52).render(message, True, WHITE)
        self.show(display, message)

    def show(self, display, message):
        pygame.draw.rect(display, DARK_GREY,
                         (WIDTH + 0.1*BAR_WIDTH, self.y, 0.8*BAR_WIDTH, self.w))
        display.blit(self.font, (WIDTH + 0.5*BAR_WIDTH, self.y))


pygame.init()
screen = pygame.display.set_mode((WIDTH + BAR_WIDTH, WIDTH))
pygame.draw.rect(screen, GREY, (WIDTH, 0, BAR_WIDTH, WIDTH))

# reset_button = Button(50, screen, 'Reset')

downsize = T.Resize((28, 28))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(INPUT_SIZE, HIDDEN_SIZE1, HIDDEN_SIZE2, OUTPUT_SIZE)
model.load_state_dict(torch.load(FILE))
model.to(device)
model.eval()


def draw(display):
    while True:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONUP:
                return

        mPos = pygame.mouse.get_pos()
        if mPos[0] > WIDTH-15:
            return
        pygame.draw.circle(display, WHITE, mPos, 13)
        pygame.draw.circle(display, LIGHT_WHITE, mPos, 20)
        pygame.display.flip()


def getImage(display, downsize, device):
    img = pygame.surfarray.array_red(display)[:WIDTH][:WIDTH]/255
    img = torch.from_numpy(img).reshape(
        (1, 28*RES, 28*RES)).transpose(1, 2).to(device)

    # plt.imshow(img[0], cmap='gray')
    # plt.figure()
    # plt.imshow(downsize(img.cpu())[0], cmap='gray')
    # plt.show()
    return downsize(img).reshape((1, 28*28)).to(torch.float32)


while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN:
            draw(screen)

        if event.type == pygame.KEYDOWN and event.key == pygame.K_DELETE:
            screen.fill(BLACK)

        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            img = getImage(screen, downsize, device)
            outputs = model(img)
            # max returns (value ,index)
            certainty, prediction = torch.max(outputs.data, 1)
            print(
                f'The number is: {prediction.item()}, with {torch.softmax(outputs, 1)[0][prediction.item()].item()*100:.1f} % certainty')

    pygame.display.flip()
