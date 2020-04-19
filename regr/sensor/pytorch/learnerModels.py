from torch import nn
import torch
import torch.nn.functional as F


class LSTMModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, bidirectional=False,
                 num_layers=1, batch_size=1):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                            bidirectional=bidirectional)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim))

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, share_hidden = self.lstm(input)

        return lstm_out


class PyTorchFC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PyTorchFC, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x = F.relu(self.fc1(x))
        x = self.fc1(x[:, -1, :])
        return F.softmax(x, dim=1)


class PyTorchFCRelu(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PyTorchFCRelu, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        # x = self.fc1(x[:, -1, :])
        return x