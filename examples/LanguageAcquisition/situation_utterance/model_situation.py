import torch
from torch import nn
import torch.nn.functional as F

from regr.program.model.pytorch import PoiModel, IMLModel
from regr.program.loss import BCEWithLogitsLoss, BCEWithLogitsIMLoss
from regr.program.metric import MacroAverageTracker, ValueTracker

device = "cpu"

class LearnerM(nn.Module):
    
    def __init__(self, num_elements, num_hidden, num_out):
        
        super().__init__()
        
        self.embedding = nn.Embedding(num_elements,num_hidden)
        self.linear = nn.Linear(num_hidden,num_out)
        
    def forward(self, *inputs):
        
        embed = self.embedding(inputs[0]).view(1,-1)
        output = self.linear(embed)
        
        return output
    

class RNNEncoder(nn.Module):
    '''
        This class creates an encoder model. This encoder is the same for the 
        situation encoder and utterance encoder used by the teacher and learner.
    '''
    
    
    def __init__(self, input_size,hidden_size):
        '''
            The init method creates the encoder model with the provided
            input_size and hidden_size.
        '''
        
        super(RNNEncoder,self).__init__()
          
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
    

class RNNDecoder(nn.Module):
    '''
        This class creates a decoder model. This decoder is used by the teacher
        and learner.
    '''
    
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=12):
        super(RNNDecoder, self).__init__()
        
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
    
    # def loss_fn():
    #     pass
    

class LearnerModel(nn.Module):
    
    
    def __init__(self, vocabulary, predicates, encoder_size=None, decoder_size=None, lr=0.0001, max_length = 12):
        '''
            What should be the parameters for the learner model?
            
            1) situation encoder
            2) situation decoder
            3) optimizers?
            4) vocabulary
            5) predicates
            6) vocabulary tensors?
            7) predicate tensors?
    
        '''
        
        super(LearnerModel, self).__init__()
        
        if encoder_size != None:
            self.encoder = RNNEncoder(*encoder_size).to(device)
        else:
            self.encoder = None
            
        if decoder_size != None:
            self.decoder = RNNDecoder(*decoder_size).to(device)
        else:
            self.decoder = None
            
            
        self.max_length = max_length
        self.load_vocabulary(vocabulary)
        self.load_predicates(predicates)
        
    def load_vocabulary(self, V):
        '''
            This method assigns the vocabulary list used by the learner model.
        '''
        
        self.vocabulary = V
    
    def load_predicates(self, P):
        '''
            This method assign the predicates used by the learner model.
        '''
        
        self.predicates = P
    
    
    def forward(self, input_seq):
        '''
            The seq2seq module's forward function receives the tensor of word indices.
            It calls the encoder-decoder structure as many times as required.
            Until it generates another sequence of word indices
            
        '''
        
        # Encoder run
        encoder_input = input_seq[0]
        input_len = input_seq.size()[0]
        
        encoder_hidden = self.encoder.initHidden()
        encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=device)
    
        
        for ei in range(input_len):    
            encoder_output, encoder_hidden = self.encoder(encoder_input, encoder_hidden)
            encoder_outputs[ei] += encoder_output[0,0]
        
        
        print("Encoder hidden:", encoder_hidden)
        
        # Decoder run
        
        decoder_input = torch.tensor([[self.predicates.index("<sos>")]],device=device)
        decoder_hidden = encoder_hidden 
        decoder_attentions = torch.zeros(self.max_length, self.max_length)
        
        out_lst = []
        
        for di in range(self.max_length):
                
            
                # Call the decoder model
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                # decoder_input = decoder_output #target_tensor[di]  # Teacher forcing
        
        
        
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
        
                if self.vocabulary[topi.item()] == "<eos>":
                    break
                else:
                    out_lst.append(topv)
        
                decoder_input = topi.squeeze().detach()
        
        # Create the tensor with the indeces
        decoder_output = torch.tensor(out_lst, device=device).view(-1,1)
        
        print("Decoder output:", decoder_output)
        
        return decoder_output #, decoder_attention[:di+1]
    
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)
    
    def get_hidden(self):
        '''
            Returns the current hidden value
        '''
        
        return self.hidden
    
    # def loss_fn(self,outputs,labels):
    #     '''
    #         Computes the negative-log likelihood function loss for the word prediction
            
    #         Parameters
    #             outputs: The output value from the forward pass
    #             labels: The target value for the evaluation
                 
    #         Returns
    #             NLLLoss object
    #     '''
        
    #     return nn.NLLLoss()(outputs,labels)    

    
   