'''
    Source File: models.py
    
    This file contains all the pytorch models used by the teacher and learner
    models.
'''

# Import the pytorch Seq2Seq models
import network_models
import pickle
#import data_loader
import random

#import statement
import torch
from torch import nn
import torch.nn.functional as F

# I have not tested using GPU with the code yet.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    '''
        This class creates an encoder model. This encoder is the same for the 
        situation encoder and utterance encoder used by the teacher and learner.
    '''
    
    
    def __init__(self, input_size,hidden_size):
        '''
            The init method creates the encoder model with the provided
            input_size and hidden_size.
        '''
        
        super(Encoder,self).__init__()
          
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(input_size,hidden_size)
        
        # Using the RNN module because it is easier to find the inverse 
        self.rnn = nn.RNN(hidden_size,hidden_size)
        
        
    def forward(self, input_seq, hidden):
        '''
            This function does the forward pass of the model.
        '''
        
        output = self.embedding(input_seq).view(1,1,-1)
        output,hidden = self.rnn(output,hidden)
        
        return output, hidden

    
    def initHidden(self):
        '''
            Creates a zero-tensor with the hidden size dimensions
        '''
        
        return torch.zeros(1, 1,self.hidden_size, device = device)
    
    def get_hidden(self):
        '''
            Returns the current hidden value
        '''
        
        return self.hidden
    
class Decoder(nn.Module):
    '''
        This class creates a decoder model. This decoder is used by the teacher
        and learner.
    '''
    
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=12):
        super(Decoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.rnn = nn.RNN(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        
        
        embedded = self.embedding(input).view(1, 1, -1)
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
        
        return output, hidden, attn_weights
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    
    def get_hidden(self):
        '''
            Returns the current hidden value
        '''
        
        return self.hidden
