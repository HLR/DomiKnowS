import torch
from torch import nn
import numpy as np

class RNNTagger(nn.Module):
    
    def __init__(self, text_field, label_field, emb_dim, rnn_size, update_pretrained=False):
        super().__init__()
        
        voc_size = len(text_field.stoi)
        self.n_labels = len(label_field)       
        
        self.embedding = nn.Embedding(voc_size, emb_dim)

        # self.embedding.weight = torch.nn.Parameter(text_field.vocab.vectors, requires_grad=update_pretrained)
        self.embedding.weight = torch.nn.Parameter(text_field.vectors, requires_grad=update_pretrained)
        

        self.rnn = nn.GRU(input_size=emb_dim, hidden_size=rnn_size, 
                          bidirectional=True, num_layers=1)

        self.top_layer = nn.Linear(2*rnn_size, self.n_labels)
 
        # self.pad_word_id = text_field.stoi[text_field.pad_token]
        ### self.pad_label_id = label_field.vocab.stoi[label_field.pad_token]
        # self.pad_label_id = text_field.stoi[text_field.pad_token]
        # self.pad_word_id = text_field.stoi['unk']
        ## self.pad_label_id = label_field.vocab.stoi[label_field.pad_token]
        # self.pad_label_id = text_field.stoi['unk']

        # self.loss = torch.nn.CrossEntropyLoss(reduction='sum')
                
    def forward(self, sentences): 
        # Compute the outputs. The shape is (max_len, n_sentences, n_labels).
        embedded = self.embedding(sentences)
        embedded = embedded.unsqueeze(0)
        rnn_out, _ = self.rnn(embedded)
        scores = self.top_layer(rnn_out)
        # pad_mask = (sentences == self.pad_word_id).float()
        # scores[:, :, self.pad_label_id] += pad_mask*10000
        
        # Flatten the outputs and the gold-standard labels, to compute the loss.
        # The input to this loss needs to be one 2-dimensional and one 1-dimensional tensor.
        scores = scores.view(-1, self.n_labels) ## torch.Size([12, 9])
        # print(scores.size())
        return scores

    # def predict(self, sentences):
    #     # Compute the outputs from the linear units.
    #     embedded = self.embedding(sentences)
    #     rnn_out, _ = self.rnn(embedded)
    #     scores = self.top_layer(rnn_out)
    #     pad_mask = (sentences == self.pad_word_id).float()
    #     scores[:, :, self.pad_label_id] += pad_mask*10000

    #     # Select the top-scoring labels. The shape is now (max_len, n_sentences).
    #     predicted = scores.argmax(dim=2)
    #     # We transpose the prediction to (n_sentences, max_len), and convert it to a NumPy matrix.
    #     return predicted.t().cpu().numpy(), scores.view(-1, scores.size(-1)).cpu().numpy()