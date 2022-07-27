### Chen Zheng 05/19/2022

# Implementing a tagger based on RNNs and a linear output unit
# Our first implementation will be fairly straightforward. We apply an RNN and then a linear output unit to predict the outputs. 
# The following figure illustrates the approach. (The figure is a bit misleading here, because we are predicting BIO labels and not part-of-speech tags, 
# but you get the idea.)

# High-quality systems that for tasks such as named entity recognition and part-of-speech tagging typically use smarter word representations, 
# for instance by taking the characters into account more carefully. We just use word embeddings.

# A small issue to note here is that we don't want the system to spend effort learning to tag the padding tokens. 
# To make the system ignore the padding, we add a large number to the output corresponding to the dummy padding tag. 
# This means that the loss values for these positions will be negligible.

# Note that we structure the code a bit differently compared to our previous implementations: 
# we compute the loss in the forward method, while previously we just computed the output in this method. 
# The reason for this change is that the CRF (see below) uses this structure, and we want to keep the implementations compatible. 
# Similarly, the predict method will convert from PyTorch tensors into NumPy arrays, in order to be compatible with the CRF's prediction method.

import torch
from torch import nn
import numpy as np
import gurobipy as gp
from gurobipy import GRB

class RNNTagger(nn.Module):
    
    def __init__(self, text_field, label_field, emb_dim, rnn_size, update_pretrained=False):
        super().__init__()
        
        voc_size = len(text_field.vocab)
        self.n_labels = len(label_field.vocab)       
        
        # Embedding layer. If we're using pre-trained embeddings, copy them
        # into our embedding module.
        self.embedding = nn.Embedding(voc_size, emb_dim)
        if text_field.vocab.vectors is not None:
            self.embedding.weight = torch.nn.Parameter(text_field.vocab.vectors, 
                                                       requires_grad=update_pretrained)

        # RNN layer. We're using a bidirectional GRU with one layer.
        self.rnn = nn.GRU(input_size=emb_dim, hidden_size=rnn_size, 
                          bidirectional=True, num_layers=1)

        # Output layer. As in the example last week, the input will be two times
        # the RNN size since we are using a bidirectional RNN.
        self.top_layer = nn.Linear(2*rnn_size, self.n_labels)
 
        # To deal with the padding positions later, we need to know the
        # encoding of the padding dummy word and the corresponding dummy output tag.
        self.pad_word_id = text_field.vocab.stoi[text_field.pad_token]
        self.pad_label_id = label_field.vocab.stoi[label_field.pad_token]
    
        # Loss function that we will use during training.
        self.loss = torch.nn.CrossEntropyLoss(reduction='sum')
        
    def compute_outputs(self, sentences):
        # The words in the documents are encoded as integers. The shape of the documents
        # tensor is (max_len, n_docs), where n_docs is the number of documents in this batch,
        # and max_len is the maximal length of a document in the batch.

        # First look up the embeddings for all the words in the documents.
        # The shape is now (max_len, n_sentences, emb_dim).        
        embedded = self.embedding(sentences)

        # Apply the RNN.
        # The shape of the RNN output tensor is (max_len, n_sentences, 2*rnn_size).
        rnn_out, _ = self.rnn(embedded)
        
        # Apply the linear output layer.
        # The shape of the output tensor is (max_len, n_sentences, n_labels).
        out = self.top_layer(rnn_out)
        
        # Find the positions where the token is a dummy padding token.
        pad_mask = (sentences == self.pad_word_id).float()

        # For these positions, we add some large number in the column corresponding
        # to the dummy padding label.
        out[:, :, self.pad_label_id] += pad_mask*10000

        return out
                
    def forward(self, sentences, labels):
        # As discussed above, this method first computes the predictions, and then
        # the loss function.
        
        # Compute the outputs. The shape is (max_len, n_sentences, n_labels).
        scores = self.compute_outputs(sentences)
        
        # Flatten the outputs and the gold-standard labels, to compute the loss.
        # The input to this loss needs to be one 2-dimensional and one 1-dimensional tensor.
        scores = scores.view(-1, self.n_labels)
        labels = labels.view(-1)
        return self.loss(scores, labels)

    def predict(self, sentences):
        # Compute the outputs from the linear units.
        scores = self.compute_outputs(sentences)

        # Select the top-scoring labels. The shape is now (max_len, n_sentences).
        predicted = scores.argmax(dim=2)
        # We transpose the prediction to (n_sentences, max_len), and convert it
        # to a NumPy matrix.
        return predicted.t().cpu().numpy(), scores.view(-1, scores.size(-1)).cpu().numpy()

    
    # def ilp_decoder(tag_matrix, label_vocab):
    #     try:
    #         label_dict = label_vocab.stoi
    #         new_label_dict = {value:key for key, value in label_dict.items()}
    #         m = gp.Model("mip1")
    #         m.setParam('OutputFlag', 0)
    #         binary_parameters = []
    #         predicates = []
    #         label_predicates = []
    #         object_sum = 0.0

    #         # build 1-0 ILP
    #         for row_num in range(tag_matrix.shape[0]):
    #             temp = []
    #             for column_num in range(tag_matrix.shape[1]):
    #                 each_parameter = m.addVar(vtype=GRB.BINARY,name = "x%s_%s" % (row_num,column_num))
    #                 object_sum += each_parameter * tag_matrix[row_num][column_num]
    #                 temp.append(each_parameter)
    #             binary_parameters.append(temp)
            
    #         # set objective
    #         m.setObjective(object_sum, GRB.MAXIMIZE)
    #         m.update() 

    #         # add constraints
    #         B_V_sum_constrint = 0
    #         arg0_sum_constraint = 0
    #         arg1_sum_constraint = 0
    #         arg2_sum_constraint = 0
    #         arg3_sum_constraint = 0

    #         B_start_label = []
    #         B_R_start_label = []
    #         all_sum_dict = {}

    #         for each_label in label_dict.keys():
    #             if "B-ARG" in each_label or "B-ARGM" in each_label:
    #                 B_start_label.append(each_label)
    #             elif "B-R" in each_label:
    #                 B_R_start_label.append(each_label)

            
    #         for each_B_start_label in B_start_label:
    #             all_sum_consrtaint = 0
    #             for row_id, each_row in enumerate(binary_parameters):
    #                 all_sum_consrtaint += each_row[label_dict[each_B_start_label]]
    #             all_sum_dict[each_B_start_label] = all_sum_consrtaint

    #         for row_id, each_row in enumerate(binary_parameters):
    #             # to make sure each token at least has one label
    #             m.addConstr(sum(each_row) == 1,name = "c%s" % row_id)
    #             # each sentence at least contain a "B-V"
    #             B_V_sum_constrint += each_row[label_dict['B-V']]

    #             arg0_sum_constraint += each_row[label_dict['B-MISC']]
    #             arg1_sum_constraint += each_row[label_dict['B-ORG']]
    #             arg2_sum_constraint += each_row[label_dict['B-LOC']]
    #             arg3_sum_constraint += each_row[label_dict['B-PER']]
                

    #             for column_id, each_column in enumerate(each_row):
    #                 # the label for the first token could not be I
    #                 if row_id == 0:
    #                     if new_label_dict[column_id][0] == "I":
    #                         m.addConstr(binary_parameters[row_id][column_id]== 0, name = "BIO%s_%s" % (row_id, column_id))
    #                 # BI constrints
    #                 if new_label_dict[column_id][0] == "I":
    #                     BIO_i_label = label_dict["B"+new_label_dict[column_id][1:]]
    #                     BIO_b_label = column_id
    #                     m.addConstr(binary_parameters[row_id-1][BIO_i_label] + binary_parameters[row_id-1][BIO_b_label] >= each_column,name = "BIO%s_%s" % (row_id, column_id))
    #                 #B-R constraints
    #                 for each_B_R in B_R_start_label:
    #                     if new_label_dict[column_id] == each_B_R:
    #                         new_each_B_R = each_B_R.split('-')
    #                         del  new_each_B_R[1]
    #                         m.addConstr(all_sum_dict["-".join(new_each_B_R)] >= each_column,name = "%s_%s_%s" % (each_B_R, row_id, column_id))
            
    
    #         m.addConstr( B_V_sum_constrint >= 1, name = "B_V_constriant")
    #         m.addConstr( all_sum_dict['B-ARG0'] <= 1, name = "arg0_constriant")
    #         m.addConstr( all_sum_dict['B-ARG1'] <= 1, name = "arg1_constriant")
    #         m.addConstr( all_sum_dict['B-ARG2'] <= 1, name = "arg2_constriant")
    #         m.addConstr( all_sum_dict['B-ARG3'] <= 1, name = "arg3_constriant")
    #         m.update()
    #         m.optimize()
    #         for each_binary_list in binary_parameters:
    #             for value_id, each_value in enumerate(each_binary_list):
    #                 if each_value.x == 1:
    #                     predicates.append(value_id)
    #                     label_predicates.append(new_label_dict[value_id])
    #     except gp.GurobiError as e:
    #         print('Error code ' + str(e.errno) + ': ' + str(e))

    #     except AttributeError:
    #         print('Encountered an attribute error')

    #     #return predicates
    #     return label_predicates, label_predicates.index('B-V')




# Implementing a conditional random field tagger
# We will now add a CRF layer on top of the linear output units. 
# The CRF will help the model handle the interactions between output tags more consistently, 
# e.g. not mixing up B and I tags of different types. Here is a figure that shows the intuition.

# The two important methods in the CRF module correspond to the two main algorithm that a CRF needs to implement:

# decode applies the Viterbi algorithm to compute the highest-scoring sequences.
# forward applies the forward algorithm to compute the log likelihood of the training set.
# Most of the code is identical to the implementation above. The differences are in the forward and predict methods.

from torchcrf import CRF

class RNNCRFTagger(nn.Module):
    
    def __init__(self, text_field, label_field, emb_dim, rnn_size, update_pretrained=False):
        super().__init__()
        
        voc_size = len(text_field.vocab)
        self.n_labels = len(label_field.vocab)       
        
        self.embedding = nn.Embedding(voc_size, emb_dim)
        if text_field.vocab.vectors is not None:
            self.embedding.weight = torch.nn.Parameter(text_field.vocab.vectors, 
                                                       requires_grad=update_pretrained)

        self.rnn = nn.GRU(input_size=emb_dim, hidden_size=rnn_size, 
                          bidirectional=True, num_layers=1)

        self.top_layer = nn.Linear(2*rnn_size, self.n_labels)
 
        self.pad_word_id = text_field.vocab.stoi[text_field.pad_token]
        self.pad_label_id = label_field.vocab.stoi[label_field.pad_token]
    
        self.crf = CRF(self.n_labels)
        
    def compute_outputs(self, sentences):
        embedded = self.embedding(sentences)
        rnn_out, _ = self.rnn(embedded)
        out = self.top_layer(rnn_out)
        
        pad_mask = (sentences == self.pad_word_id).float()
        out[:, :, self.pad_label_id] += pad_mask*10000
        
        return out
                
    def forward(self, sentences, labels):
        # Compute the outputs of the lower layers, which will be used as emission
        # scores for the CRF.
        scores = self.compute_outputs(sentences)

        # We return the loss value. The CRF returns the log likelihood, but we return 
        # the *negative* log likelihood as the loss value.            
        # PyTorch's optimizers *minimize* the loss, while we want to *maximize* the
        # log likelihood.
        return -self.crf(scores, labels)
            
    def predict(self, sentences):
        # Compute the emission scores, as above.
        scores = self.compute_outputs(sentences)

        # Apply the Viterbi algorithm to get the predictions. This implementation returns
        # the result as a list of lists (not a tensor), corresponding to a matrix
        # of shape (n_sentences, max_len).
        return self.crf.decode(scores)