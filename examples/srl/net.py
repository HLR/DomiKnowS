import torch.nn as nn
import torch


class SimpleLSTM(nn.Module):
    def __init__(self,
                 num_labels=3,
                 predicate_size=100,
                 hidden_size=300,
                 recurrent_dropout=0.0,
                 token_size=300,
                 num_layers=2):
        super().__init__()

        self.num_labels = num_labels
        self.token_size = token_size
        self.predicate_size = predicate_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.predicate_embedding = nn.Embedding(2, self.predicate_size)

        self.bilstm = nn.LSTM(input_size=self.token_size + self.predicate_size,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              bidirectional=True)

        self.proj = nn.Linear(self.hidden_size * 2, self.num_labels)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, token_embeds, predicate_labels):
        # token_embeds: (seq_length, 1, 300)
        # predicate_labels: (seq_length, 1, 1)

        token_embeds = token_embeds.unsqueeze(1)
        predicate_embeds = predicate_labels.unsqueeze(1)

        predicate_embeds = self.predicate_embedding(predicate_labels)  # (seq_length, 1, predicate_size)

        input_embeds = torch.cat((token_embeds, predicate_embeds), dim=2)

        h_out, _ = self.bilstm(input_embeds)  # seq_length, 1, 2 * hidden_size

        output = self.relu(h_out)
        out_shape = output.shape

        logits = self.proj(output.squeeze(1))  # seq_length, num_labels

        return logits
