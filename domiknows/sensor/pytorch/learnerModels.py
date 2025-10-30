from torch import nn
import torch
import torch.nn.functional as F


class LSTMModel(nn.Module):
    """
    Torch module for an LSTM.
    """
    def __init__(self, input_dim, hidden_dim, bidirectional=False,
                 num_layers=1, batch_size=1):
        """
        Initializes the LSTM module.

        Args:
        - input_dim: Feature size of each input timestep.
        - hidden_dim: Number of features in the hidden state.
        - bidirectional (optional): If True, use a bidirectional LSTM. Defaults to False.
        - num_layers (optional): Number of stacked LSTM layers. Defaults to 1.
        - batch_size (optional): Batch size used for hidden state init. Defaults to 1.
        """
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                            bidirectional=bidirectional)

    def init_hidden(self):
        """
        Creates the zero-initialized hidden state and cell state.

        Returns:
        - Tuple of tensors corresponding to the initial hidden state and cell state.
            Each tensor has shape `(num_layers * 2, batch_size, hidden_dim)` filled
            with zeros.
        """
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim))

    def forward(self, input):
        """
        Runs the LSTM over a sequence.

        Args:
        - input: Input of shape `(seq_len, batch_size, input_dim)`.

        Returns:
        - Output features from the last layer of the LSTM for each timestep.
            Has shape `(seq_len, batch_size, hidden_dim * num_directions)`.
        """
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, share_hidden = self.lstm(input)

        return lstm_out


class PyTorchFC(nn.Module):
    """
    Torch module for a fully-connected layer over sequence inputs with softmax activations.
    """
    def __init__(self, input_dim, output_dim):
        """
        Initializes the FC with softmax module.

        Args:
        - input_dim: Feature dimension of the last timestep.
        - output_dim: Number of output classes.
        """
        super(PyTorchFC, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Compute class probabilities from a linear layer for the last timestep in
        the sequence input.

        Args:
        - x: Sequence batch of shape `(batch_size, seq_len, input_dim)`
            The last timestep: `x[:, -1, :]` is used as input.

        Returns:
        - Softmax probabilities of shape `(batch_size, output_dim)`.
        """
        # x = F.relu(self.fc1(x))
        x = self.fc1(x[:, -1, :])
        return F.softmax(x, dim=1)


class PyTorchFCRelu(nn.Module):
    """
    Torch module for a fully-connected layer with LeakyReLU activations.
    """
    def __init__(self, input_dim, output_dim):
        """
        Initialize the FC with LeakyReLU layer.

        Args:
            input_dim: Input feature dimension.
            output_dim: Output feature dimension.
        """
        super(PyTorchFCRelu, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Apply a linear layer followed by LeakyReLU.

        Args:
        - x: Input tensor whose last dimension equals `input_dim`
            Leading dimensions are preserved.

        Returns:
        - Activated features with the same leading dimensions as
            the input tensor and last dimension `output_dim`.
        """
        x = F.leaky_relu(self.fc1(x))
        # x = self.fc1(x[:, -1, :])
        return x