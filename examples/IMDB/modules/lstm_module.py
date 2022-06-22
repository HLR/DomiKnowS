import sys
sys.path.append("../../../")

from torch import nn
import torch

class LSTMModule(nn.Module):
  def __init__(self, embed_size, hidden_size, drop_rate):
    super(LSTMModule, self).__init__()

    # hyperparameters
    self.embed_size = embed_size
    self.hidden_size = hidden_size
    self.drop_rate = drop_rate

    # modules
    self.rnn = nn.LSTM(self.embed_size, self.hidden_size, bidirectional=True)
    self.dropout = nn.Dropout(p=self.drop_rate)

  def forward(self, input):
    output, (h, c) = self.rnn(input)

    # concatenate the last outputs of the forward and backward LSTMs
    forward, backward = torch.chunk(output, 2, dim=2)
    comb = torch.cat((forward[-1,:,:], backward[0,:,:]), dim=1)

    model_out = self.dropout(comb)

    return model_out