import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Constants
LEARNING_RATE = 0.1
N_EPOCHS = 100

# Data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X = torch.tensor([[0], [1], [2], [3], [4]], dtype=torch.float32, device=device)
Y = torch.tensor([[1], [3], [5], [7], [9]], dtype=torch.float32, device=device)
test = torch.tensor([[10]], dtype=torch.float32, device=device)
nSamples, nFeatures = X.shape

# Model


class Model(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim,
                                bias=True, dtype=torch.float32, device=device)

    def forward(self, x) -> torch.Tensor:
        return self.linear(x)


model = Model(nFeatures, 1)


# Optimizer ans loss
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
loss = nn.MSELoss()

# Training
for epoch in range(N_EPOCHS):
    prediction = model(X)

    error = loss(prediction, Y)

    error.backward()

    optimizer.step()
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        w, b = model.parameters()
        print(w.item(), b.item(), model(test).item())

# Validation
with torch.no_grad():
    print(model(test).item())

    plt.figure()
    plt.plot(X.cpu(), Y.cpu(), 'r*')
    plt.plot(X.cpu(), model(X).cpu())
    plt.show()
