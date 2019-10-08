from Graphs.Learners.mainLearners import CallingLearner
from torch import nn
from typing import Dict, Any
import torch
import pdb

class LSTMFlair(nn.Module):

    def __init__(self, input_dim, hidden_dim, bidirectional=False,
                 num_layers=1, batch_size=1):
        super(LSTMFlair, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, bidirectional=bidirectional)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers*2, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers*2, self.batch_size, self.hidden_dim))

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, share_hidden = self.lstm(input)

        return lstm_out


class LSTMLearner(CallingLearner):
    def __init__(self, *pres, input_dim, hidden_dim, num_layers=1, bidirectional=False):
        super(LSTMLearner, self).__init__(*pres)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.input_dim = input_dim
        self.model = LSTMFlair(input_dim=self.input_dim, hidden_dim=self.hidden_dim, num_layers=self.num_layers, batch_size=1, bidirectional=self.bidirectional)

    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        super(LSTMLearner, self).forward(context=context)
        print(context)
        _list = []
        for token in context[self.pres[0].fullname]:
            _list.append(token.embedding.view(1, self.input_dim))
        _tensor = torch.stack(_list)
        output = self.model(_tensor)
        pdb.set_trace()
        return output


class FullyConnectedLearner(CallingLearner):
    pass





