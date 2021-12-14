# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 18:50:08 2021

@author: Juan
"""
import sys
import torch

if '../../..' not in sys.path:
    sys.path.append('../../..')
print("sys.path - %s"%(sys.path))


from reader_situation import InteractionReader
from model_situation import LearnerModel
from regr.program.loss import NBCrossEntropyLoss

# Use functional sensor instead
def Situation_Convert(*inputs):
    
    predicates = [x.strip() for x in open("../data/predicates.txt")]
    predicates.append("<eos>")
    
    max_len = 12
    
    situation = inputs[0][0]
    indices = [predicates.index(logic) for logic in situation]
    tensor = torch.tensor(indices,dtype=torch.long, device='cpu')
    
    print("Indices:", indices)
    print("Situation Tensor:", tensor)
    
    return tensor.view(-1,1)

def Utterance_Convert(*inputs):
    
    vocabulary = [x.strip() for x in open("../data/vocabulary.txt")]
    vocabulary.insert(0, "<sos>")
    
    max_len = 12
    
    
    utterance = inputs[0][0]
    indices = [vocabulary.index(word) for word in utterance]
    tensor = torch.tensor(indices,dtype=torch.long, device='cpu')
    
    print("Indices: ", indices)
    print("Utterance Tensor: ", tensor)
    
    return tensor.view(-1,1)

def train(train_data,model):
    
    # Initialize the loss function
    criterion = NBCrossEntropyLoss()
    loss = 0.0
    
    # Get the data running
    
    
    for ex in list(iter(train_data)):
        
        print(ex)
        
        situation = ex['situation']
        utterance = ex['utterance']
        
        # Conversion test
        sit_emb = Situation_Convert(situation)
        utt_emb = Utterance_Convert(utterance)
        
        print(sit_emb)
        print(utt_emb)
        
        # Check if the learner module accepts the input
        output = model.forward(sit_emb)
        
        print("U embed:",utt_emb)
        print("Output:", output)
        
        # Apply the loss function
        loss += criterion(output,utt_emb)
            
        print(loss)

def main():
    '''
        This code is to test the following:
            1) Data is read in the correct format.
            2) The conversion process is correctly done.
        
    '''
    
    
    # Load the training file
    train_filename = "../data/training_set.txt"
    test_filename = "../data/test_set.txt"
    
    # Get the list of predicates and words
    words = [x.strip() for x in open("../data/vocabulary.txt")]
    
    # Build the data iterators
    train_dataset = InteractionReader(train_filename,"txt")
    test_dataset = InteractionReader(train_filename,"txt")

    #device options are 'cpu', 'cuda', 'cuda:x', torch.device instance, 'auto', None
    device = 'cpu'
    
    # Instantiate learner model
    encoder_dim = (36,100)
    decoder_dim = (100,24)
    learning_rate = 0.001
    max_length = 12
    vocabulary = [x.strip() for x in open("../data/vocabulary.txt")]
    predicates = [x.strip() for x in open("../data/predicates.txt")]
    
    model = LearnerModel(vocabulary, predicates, encoder_dim, decoder_dim, learning_rate, max_length)
    
    # train the model
    train(train_dataset, model)

    
    
if __name__ == "__main__":
    main()
    # module_test()
