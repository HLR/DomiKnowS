import torch
from torch import nn
import torch.nn.functional as F


class Net(torch.nn.Module):
    def __init__(self, w=None):
        super().__init__()
        if w is not None:
            self.w = torch.nn.Parameter(torch.tensor(w).float().view(1, 2))
        else:
            self.w = torch.nn.Parameter(torch.randn(1, 2))

    def forward(self, x):
        return x.matmul(self.w)


class LearnerM(nn.Module):
    
    def __init__(self, num_elements):
        
        super().__init__()
        
        self.embedding = nn.Embedding(num_elements,20)
        self.linear = nn.Linear(20,2)
        
    def forward(self, *inputs):
        
        embed = self.embedding(inputs[0]).view(1,-1)
        output = self.linear(embed)
        
        output = F.log_softmax(output, dim=1)
        
        return output
    


class LearnerModel(nn.Module):
    
    
    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.1, max_length=12):
        super().__init__()
        
        
        # Input encoder model
        
        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.encoder_embedding = nn.Embedding(self.input_size,self.hidden_size)
        
        # Using the RNN module because it is easier to find the inverse 
        self.encoder_rnn = nn.RNN(self.hidden_size,self.hidden_size)
        
        
        
        # Output encoder model
        
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.rnn = nn.RNN(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq, hidden, encoder_outputs):
        
        # Encoder run
        input_seq = input_seq[0]
        output = self.embedding(input_seq).view(1,1,-1)
        output,hidden = self.rnn(output,hidden)
        
        
        
        # Decoder run
        input_seq = hidden
        
        embedded = self.embedding(input_seq).view(1, 1, -1)
        embedded = self.dropout(embedded)
        
    
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        
        combined_hidden = encoder_outputs
        
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), combined_hidden.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.rnn(output, hidden)

        # If I return this log softmax of the values, then I need to have
        # NLLLoss as the loss function used by this decoder.
        output = F.log_softmax(self.out(output[0]), dim=1)
        
        return output #, hidden, attn_weights
    
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)
    
    def get_hidden(self):
        '''
            Returns the current hidden value
        '''
        
        return self.hidden

    
    