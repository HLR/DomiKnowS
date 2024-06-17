import torch.nn as nn
import torch
import torch.utils.data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from viterbi import viterbi,  make_transition_matrix
import torch.nn.functional as F

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

        token_embeds = token_embeds.squeeze(0).unsqueeze(1)
        predicate_embeds = predicate_labels.squeeze(0).unsqueeze(1)

        predicate_embeds = self.predicate_embedding(predicate_labels.squeeze(0))  # (seq_length, 1, predicate_size)

        input_embeds = torch.cat((token_embeds, predicate_embeds), dim=2)

        h_out, _ = self.bilstm(input_embeds)  # seq_length, 1, 2 * hidden_size

        output = self.relu(h_out)
        out_shape = output.shape

        logits = self.proj(output.squeeze(1))  # seq_length, num_labels

        logits = logits.unsqueeze(0) # (1, seq_length, num_labels)

        #return torch.tensor([[0, 1, 0],
        #                     [1, 0, 0],
        #                     [1, 0, 0],
        #                     [1, 0, 0]]).float().unsqueeze(0)

        return logits

def flipBatch(data, lengths):
    # https://stackoverflow.com/questions/55904997/pytorch-equivalent-of-tf-reverse-sequence
    # batch is dim 1
    assert data.shape[1] == len(lengths), "Dimension Mismatch!"
    data = torch.clone(data)
    for i in range(data.shape[1]):
        data[:lengths[i], i] = data[:lengths[i], i].flip(dims=[0])

    return data

class HighwayLSTM(nn.Module):
    def __init__(self,
                 label_space,
                 predicate_size = 100,
                 hidden_size = 300,
                 dropout = 0.1,
                 token_size = 300,
                 num_layers = 4):
        
        super().__init__()
    
        self.label_space = label_space
        self.num_labels = len(self.label_space)
        self.token_size = token_size
        self.predicate_size = predicate_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.predicate_embedding = nn.Embedding(2, self.predicate_size)
        #self.token_embedding = nn.Embedding.from_pretrained(glove_vectors.vectors, freeze=True)
        
        self.lstm_layers = nn.ModuleList([nn.LSTM(input_size = self.token_size + self.predicate_size, 
                                                 hidden_size = self.hidden_size,
                                                 num_layers = 1,
                                                 bidirectional = False)])
        self.gate_layers = nn.ModuleList()
        self.resid_layers = nn.ModuleList()
        self.sigmoid = nn.Sigmoid()
        
        for _ in range(self.num_layers - 1):
            self.lstm_layers.append(nn.LSTM(input_size = self.hidden_size, 
                                            hidden_size = self.hidden_size,
                                            num_layers = 1,
                                            bidirectional = False))
            
            self.gate_layers.append(nn.Linear(self.hidden_size + self.hidden_size, 1))
            self.resid_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
        
        self.proj = nn.Linear(self.hidden_size, self.num_labels)
        
        self.dropout = nn.Dropout(p = dropout)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.transition_matrix = make_transition_matrix(self.label_space)
        
    def forward(self, token_embeds, predicate_labels):
        token_embeds = token_embeds.permute(1, 0, 2)
        predicate_labels = predicate_labels.permute(1, 0, 2)

        # token_embeds: (seq_length, batch_size, 300) - packed sequence
        # predicate_labels: (seq_length*, batch_size, 1) - packed sequence

        token_lens = [token_embeds.shape[0]]

        predicate_embeds = self.predicate_embedding(predicate_labels).squeeze(1) # (seq_length*, batch_size, predicate_size)
        
        input_embeds = torch.cat((token_embeds, predicate_embeds), dim=2)
        
        x = input_embeds
        prev_output = None
        for layer_idx in range(self.num_layers):
            lstm = self.lstm_layers[layer_idx]

            # go in reverse on odd dimensions
            if layer_idx != 0:
                x = flipBatch(x, token_lens)
            
            # save input for highway connection/residual
            flipped_input = x
            
            # lstm layer forward
            x = pack_padded_sequence(x, token_lens, batch_first=False, enforce_sorted=False)
            h_out, _ = lstm(x) # seq_length, batch_size, hidden_size
            h_out, _ = pad_packed_sequence(h_out)
            #h_out = self.relu(h_out) # longest, batch_size, hidden_size
            x = h_out
            
            # highway/residual connection
            if layer_idx != 0:
                h_shift = torch.roll(h_out, 1, dims=0)
                h_shift[0, :, :] = 0

                gate_input = torch.cat([h_shift, flipped_input], dim=2)
                gate_input_shape = gate_input.shape
                gate_input = gate_input.reshape(-1, gate_input_shape[-1])
                gate_out = self.sigmoid(self.gate_layers[layer_idx - 1](gate_input))
                gate_out = gate_out.reshape(*gate_input_shape[:2], 1)
                
                flip_prev = flipBatch(prev_output, token_lens)
                resid_shape = flip_prev.shape
                proj_resid = self.resid_layers[layer_idx - 1](flip_prev.reshape(-1, resid_shape[-1]))
                proj_resid = proj_resid.reshape(*resid_shape)
                
                x = gate_out * x + (1 - gate_out) * proj_resid
                
                #x = x + flipBatch(prev_output, token_lens)
            
            # dropout
            x = self.dropout(x)
            
            # save prev output
            prev_output = h_out

        output_padded = x
            
        out_shape = output_padded.shape
        
        logits = self.proj(output_padded.view(out_shape[0] * out_shape[1], out_shape[2]))
        logits = logits.view(out_shape[0], out_shape[1], -1) # batch_size, longest, num_labels
        
        logits = logits[:, 0, :] # longest, num_labels

        # viterbi decode bio tags
        tags, scores = viterbi(logits, self.transition_matrix)

        # get likelihood of best path
        logprobs = F.log_softmax(logits, dim=-1)
        #logits = torch.ones((logits.shape[0], 5)) * -10
        logits = torch.ones((logits.shape[0], 3)) * -10

        # convert path to aggregate likelihood and logits
        likelihood = 0.0
        for i in range(logits.shape[0]):
            likelihood += logprobs[i, tags[i]]

            if tags[i] == 0 or tags[i] == 1:
                logits[i, 1] = 10
            elif tags[i] == 2 or tags[i] == 3:
                logits[i, 2] = 10

        '''
        # ILP only
        # sum probabilities, don't use viterbi
        probs = F.softmax(logits, dim=1)

        aggr_matrix = torch.tensor(
            [
                [0,1,0],
                [0,1,0],
                [0,0,1],
                [0,0,1],
                [1,0,0],
            ]
        ).float()

        logits = torch.mm(probs, aggr_matrix)

        logits = torch.log(logits)
        '''

        return [[logits.unsqueeze(0), likelihood]]
