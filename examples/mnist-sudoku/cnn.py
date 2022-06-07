from torch import nn
import torch

class Net(torch.nn.Module):
    def __init__(self, size):
        super().__init__()

        self.size = size

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.lin1 = nn.Linear(256, 128)
        self.lin2 = nn.Linear(128, 9)

        self.relu = nn.ReLU()

        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.reshape(self.size ** 2, 1, 28, 28)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.reshape(self.size ** 2, -1)

        x = self.lin1(x)
        x = self.relu(x)

        x = self.drop(x)

        y_digit = self.lin2(x)

        return torch.unsqueeze(y_digit, dim=0)
