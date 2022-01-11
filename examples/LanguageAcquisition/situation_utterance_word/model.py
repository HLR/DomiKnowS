import torch
from torch import nn
import torch.nn.functional as F

from regr.program.model.pytorch import PoiModel, IMLModel
from regr.program.loss import BCEWithLogitsLoss, BCEWithLogitsIMLoss
from regr.program.metric import MacroAverageTracker, ValueTracker

device = "cpu"

class HeadLayer(nn.Module):

    def __init__(self, hidden_dim, target_size):
        super(HeadLayer, self).__init__()
        self.linear_layer = nn.Linear(hidden_dim, target_size)

    def forward(self, x):
        #t=time.time()
        out = self.linear_layer(x)
        #print("linear time: ",time.time()-t)
        return out

class GenerateLabel():
    
    def __init__(self,device,vocabulary):
        self.device=device
        self.vocabulary=vocabulary

    def __call__(self, label):

        label_list=[]
        
        print("Label:", label)
        
        # read the max number of words
        c = 0
        for i in label:
            try:

                if i == "<eos>" and c != 0:
                    label_list.append(self.vocabulary.index(i))
                elif i == '<eos>' and c == 0:
                    label_list.append(self.vocabulary.index(i))
                    c += 1
                else:
                    label_list.append(self.vocabulary.index(i))


            except:
                label_list.append(-100)
                
        
        output = torch.LongTensor(label_list).to(self.device)
        
        # output = output.unsqueeze(0)
        
        print("Output Labels:", output)
        print("Output size:", output.size())
        return output

class LearnerRNN(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, pred_size, predicates, device):
        super(LearnerRNN, self).__init__()
        self.device=device
        self.predicates=predicates
        self.word_embeddings = nn.Embedding(pred_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)

    def forward(self, situation_indices):

        print("Situation indices:", situation_indices)
        embeds = self.word_embeddings(situation_indices.to(self.device))

        rnn_out, _ = self.rnn(embeds)

        return rnn_out #[1:-1]
    

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
        output = self.out(output[0]) # Apply Linear Layer before softmax
        # output = F.log_softmax(output, dim=1)
        
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
        input_len = encoder_input.size()[0]
        
        print("Encoder Input:", encoder_input)
        print("Input length:", input_len)
        
        print("-"*40)
        
        encoder_hidden = self.encoder.initHidden()
        encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=device)
    
        
        for ei in range(input_len):    
            encoder_output, encoder_hidden = self.encoder(encoder_input[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0,0]
        
        
        # print("Encoder hidden:", encoder_hidden.size())
        
        # Decoder run
        
        decoder_input = torch.tensor([[self.predicates.index("<sos>")]],device=device)
        decoder_hidden = encoder_hidden 
        decoder_attentions = torch.zeros(self.max_length, self.max_length)
        
        out_lst = []
        
        # Need to find a way to use the decoder output with the grad function
        # and requires grad.
        # The model decodes each word from the model, how to use autograd for 
        # the new tensor with the the same gradients.
        
        for di in range(self.max_length):
                
            
                # Call the decoder model
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                # decoder_input = decoder_output #target_tensor[di]  # Teacher forcing
        
        
        
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
        
                # print("Decoder output grad function: ", decoder_output.)
                
                # if not(topi.item() < len(self.predicates)):
                #     print("Item results:", topi.item(), len(self.vocabulary), self.vocabulary)
                
                if self.vocabulary[topi.item()] == "<eos>":

                   # Need to pad the decoder output with "<eos>" before stopping
                   out_lst.append(decoder_output)



                   diff = self.max_length - len(out_lst)


                   for k in range(diff):
                       # Add a zero vector when we break the loop.

                       p = [0. for number in range(23)]

                       #p = torch.zeros(1,23, requires_grad=True)
                       p[22] = 3.0 # Magic number to determine that the last element is to be selected.

                       out_p = torch.tensor(p, requires_grad=True)

                       out_lst.append(out_p.unsqueeze(0))

                   break

                else:
                    # Append the probability distribution of the output data
                    # So that DomiKnows loss function does CE loss
                    out_lst.append(decoder_output)
                    # out_lst.append(topi.item())
        
                decoder_input = topi.squeeze().detach()
                
        # print("output list:", out_lst)
        
        # Create the tensor with the indeces
        # decoder_output = torch.tensor(out_lst, device=device).view(-1,1)
        
        ###################################################################
        # Need to use torch.cat for this output.
        
        # Create a tensor with each element of output list
        # final_tensor = torch.zeros( (len(out_lst), *out_lst[0].size()) )
        
        # # Replace each zero tensor with the tensor from the model's output.
        # for i,e in enumerate(out_lst):
        #     final_tensor[i] = e
            
        # decoder_output = final_tensor
        
        decoder_output = torch.cat(out_lst)
        
        # print("Decoder output size:", decoder_output.size())
        
        decoder_output.reshape((len(out_lst),len(self.vocabulary)))
        
        # print("Decoder output size (After reshape):", decoder_output.size())
        # print("Decoder output:", decoder_output)
        
        # decoder_output = torch.tensor(out_lst)
        
        print("ModuleLearner Output:", decoder_output)
        
        return decoder_output #.unsqueeze(0) #, decoder_attention[:di+1]
    
    
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

    
   