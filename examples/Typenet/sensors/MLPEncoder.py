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
        self.lin2 = nn.Linear(hidden_dim*2, hidden_dim)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, sentences, mention_rep):
       # mention_rep: (batch_size * bag_size, embedding_dim)
        mention_rep = mention_rep.to(device=config.device)

        bag_size = len(sentences[0])

        t0 = time.time()
        # sentences: (batch_size, bag_size, sentence_length [variable])
        embed_bag = torch.empty((len(sentences) * bag_size, self.embedding_shape[-1]), device=config.device)

        for batch_idx, item in enumerate(sentences): # iter through batch
            for sent_idx, sent in enumerate(item): # iter through bag
                sent_tensor = torch.tensor(sent, dtype=int, device=config.device)

                sentence_embed = self.embeddings(sent_tensor)

                embed_bag[(batch_idx * bag_size) + sent_idx] = torch.mean(sentence_embed, dim=0)

        t1 = time.time()
        lin1_out = self.lin1(torch.cat((embed_bag, mention_rep.view(mention_rep.shape[0]*mention_rep.shape[1], -1)), dim=1).float())
        lin1_out = self.relu(lin1_out)
        lin1_out = self.dropout(lin1_out)

        lin2_out = self.lin2(lin1_out)
        lin2_out = self.relu(lin2_out)

        if config.timing:
            print('MLPEncoder', time.time() - t1, t1 - t0)

        return lin2_out