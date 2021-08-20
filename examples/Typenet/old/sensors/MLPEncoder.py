import torch
from torch import nn
import config
import time

class MLPEncoder(nn.Module):
    def __init__(self, pretrained_embeddings, mention_dim, hidden_dim=128):
        super(MLPEncoder, self).__init__()
        
        self.embedding_shape = pretrained_embeddings.shape

        self.embeddings = nn.Embedding(*self.embedding_shape)
        self.embeddings.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        self.embeddings.weight.requires_grad = False
    
        self.lin1 = nn.Linear(mention_dim + self.embedding_shape[-1], hidden_dim*2)
        self.lin2 = nn.Linear(hidden_dim*2, config.num_types)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, sentences, mention_rep):
        mention_rep = mention_rep.to(device=config.device)

        embed_bag = torch.empty((len(sentences), self.embedding_shape[-1]))
                        
        for i, sent in enumerate(sentences):
            sent_tensor = torch.tensor(sent, dtype=int).to(device=config.device)

            sentence_embed = self.embeddings(sent_tensor)

            embed_bag[i] = torch.mean(sentence_embed, dim=0)

        lin1_out = self.lin1(torch.cat((embed_bag, mention_rep), dim=1).float()) #torch.cat((embed_bag, mention_rep), dim=1).float()
        lin1_out = self.relu(lin1_out)
        
        lin2_out = self.lin2(lin1_out)

        #print('time MLPEncoder', time.time() - t1)

        return lin2_out