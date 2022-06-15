import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys

# Constants
FILE = 'mnist_model.pth'
LEARNING_RATE = 0.0005
N_EPOCHS = 100
BATCH_SIZE = 100

INPUT_SIZE = 28*28
OUTPUT_SIZE = 10
HIDDEN_SIZE1 = 512
HIDDEN_SIZE2 = 256


class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim1,
                               bias=True, dtype=torch.float32)
        self.hidden = nn.Linear(hidden_dim1, hidden_dim2,
                                bias=True, dtype=torch.float32)
        self.output = nn.Linear(hidden_dim2, output_dim,
                                bias=True, dtype=torch.float32)

    def forward(self, x) -> torch.Tensor:
        out = F.relu(self.input(x))
        out = F.relu(self.hidden(out))
        out = self.output(out)
        return out


def loadData():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainingData = torchvision.datasets.MNIST(
        root='~/skola/egen/pytorch/MNIST/', download=True, transform=transforms.ToTensor(), train=True)
    validationData = torchvision.datasets.MNIST(
        root='~/skola/egen/pytorch/MNIST/', download=True, transform=transforms.ToTensor(), train=False)

    trainingData = torch.utils.data.DataLoader(
        trainingData, batch_size=BATCH_SIZE, shuffle=True)
    validationData = torch.utils.data.DataLoader(
        validationData, batch_size=BATCH_SIZE, shuffle=False)

    return device, trainingData, validationData


def exampleData(data):
    examples = iter(data)
    example_data, example_targets = examples.next()

    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(example_data[i][0], cmap='gray')
    plt.show()


def train(model, data, optimizer, loss, device):
    for epoch in range(N_EPOCHS):
        for i, (images, labels) in enumerate(data):
            images = images.view(-1, INPUT_SIZE).to(device)
            labels = labels.to(device)

            predictions = model(images)

            error = loss(predictions, labels)

            error.backward()

            optimizer.step()
            optimizer.zero_grad()

        print(f'Epoch [{epoch+1}/{N_EPOCHS}], Loss: {error.item():.8f}')


def validate(model, data, device):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in data:
            images = images.reshape(-1, INPUT_SIZE).to(device)
            labels = labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(
            f'Accuracy of the network on the {n_samples} test images: {acc} %')


def main():
    device, trainingData, validationData = loadData()

    model = Model(INPUT_SIZE, HIDDEN_SIZE1,
                  HIDDEN_SIZE2, OUTPUT_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss = nn.CrossEntropyLoss()

   # train(model, trainingData, optimizer, loss, device)
    exampleData(trainingData)
    validate(model, validationData, device)

    # torch.save(model.state_dict(), FILE)

    # model.load_state_dict(torch.load(FILE))
    # mode.to(device)
    # model.eval()


if __name__ == '__main__':
    main()
