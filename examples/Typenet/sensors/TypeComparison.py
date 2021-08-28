from torch import nn
import torch
import time

class TypeComparison(nn.Module):
    def __init__(self, type_embed_dim, num_types):
        super(TypeComparison, self).__init__()
        
        #self.type_embeddings = nn.Embedding(num_types, type_embed_dim)
        
        self.lin = nn.Linear(type_embed_dim, num_types)

    def log_sum_exp(self, _tensor):
        '''
           :_tensor is a (batch_size, scores, num_bags) sized torch Tensor
           return another tensor ret of size (batch_size, scores) wh    ere ret[i][j] = logsumexp(ret[i][:][j] or _tensor[i][j][:])
        '''
        max_scores, _ = torch.max(_tensor, dim = -1) # (batch_size, scores)
        return max_scores + torch.log(torch.sum(torch.exp(_tensor - max_scores.unsqueeze(-1)), dim = -1)) #(batch_size, scores)

    def forward(self, encoded):
        #type_matrix = self.type_embeddings.weight
        #dot_compare = torch.matmul(encoded, torch.transpose(type_matrix, 0, 1)) # (bag_size x embed_dim) * (embed_dim x num_types)
        #logits = self.log_sum_exp(dot_compare.unsqueeze(0).transpose(1,2))

        logits = self.lin(encoded)
        logits = self.log_sum_exp(logits.unsqueeze(0).transpose(1,2))

        return logits