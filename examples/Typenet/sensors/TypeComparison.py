from torch import nn
import torch
import time

import config

class TypeComparison(nn.Module):
    def __init__(self, type_embed_dim, num_types):
        super(TypeComparison, self).__init__()
        
        #self.type_embeddings = nn.Embedding(num_types, type_embed_dim)

        self.num_types = num_types

        self.lin = nn.Linear(type_embed_dim, self.num_types)

    def log_sum_exp(self, _tensor):
        '''
           :_tensor is a (batch_size, scores, num_bags) sized torch Tensor
           return another tensor ret of size (batch_size, scores) where ret[i][j] = logsumexp(ret[i][:][j] or _tensor[i][j][:])
        '''
        max_scores, _ = torch.max(_tensor, dim = -1) # (batch_size, scores)
        return max_scores + torch.log(torch.sum(torch.exp(_tensor - max_scores.unsqueeze(-1)), dim = -1)) #(batch_size, scores)

    def forward(self, encoded):
        # type embedding comparison
        #type_matrix = self.type_embeddings.weight
        #dot_compare = torch.matmul(encoded, torch.transpose(type_matrix, 0, 1)) # (bag_size x embed_dim) * (embed_dim x num_types)
        #logits = self.log_sum_exp(dot_compare.unsqueeze(0).transpose(1,2))

        # linear layer
        # encoded: (batch_size * bag_size, encoded_dim)
        logits = self.lin(encoded)
        # logits: (batch_size * bag_size, num_types) -> (batch_size, bag_size, num_types)
        logits = logits.reshape(config.batch_size, logits.shape[0]//config.batch_size, self.num_types)
        # logits: (batch_size, bag_size, num_types) -> (batch_size, num_types, bag_size)
        logits = logits.permute(0, 2, 1)

        logits = self.log_sum_exp(logits)

        #print(logits)

        return logits