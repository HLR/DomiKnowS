import torch
from torch import nn
import torch.nn.functional as F

from regr.program.model.pytorch import PoiModel, IMLModel
from regr.program.loss import BCEWithLogitsLoss, BCEWithLogitsIMLoss
from regr.program.metric import MacroAverageTracker, ValueTracker

class LearnerM(nn.Module):
    
    def __init__(self, num_elements, num_hidden, num_out):
        
        super().__init__()
        
        self.embedding = nn.Embedding(num_elements,num_hidden)
        self.linear = nn.Linear(num_hidden,num_out)
        
    def forward(self, *inputs):
        
        embed = self.embedding(inputs[0]).view(1,-1)
        output = self.linear(embed)
        
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
        self.max_length = 1 # max_length

        self.embedding = nn.Embedding(self.hidden_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.rnn = nn.RNN(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        
        # Encoder run
        
        # init hidden
        hidden = self.initHidden()
        
        input_seq = input_seq[0]
        output = self.embedding(input_seq).view(1,1,-1)
        output,hidden = self.rnn(output,hidden)
        
        encoder_outputs = torch.zeros(1,self.hidden_size)
        encoder_outputs[0] += output[0,0]
        
        
        
        # Decoder run
        input_seq = hidden
        
        embedded = input_seq #self.embedding(input_seq).view(1, 1, -1)
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

    
    