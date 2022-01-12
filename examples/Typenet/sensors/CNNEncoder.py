import torch
from torch import nn
import torch.nn.functional as F
import config
import time

class CNNEncoder(nn.Module):
    def __init__(self, pretrained_embeddings):
        super(CNNEncoder, self).__init__()
        
        self.embedding_shape = pretrained_embeddings.shape

        self.embeddings = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_embeddings))
        self.embeddings.weight.requires_grad = False

        self.cnn = nn.Conv1d(self.embedding_shape[-1],
                             self.embedding_shape[-1],
                             kernel_size=5,
                             stride=1)

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.tanh = nn.Tanh()

        self.dropout = nn.Dropout(0.5)

        self.lin1 = nn.Linear(self.embedding_shape[-1], self.embedding_shape[-1])
        self.lin2 = nn.Linear(self.embedding_shape[-1] * 2, self.embedding_shape[-1])
        self.lin3 = nn.Linear(self.embedding_shape[-1], self.embedding_shape[-1])

    def forward(self, sentences, mention_rep):
        # mention_rep: (batch_size * bag_size, embedding_dim)
        mention_rep = mention_rep.to(device=config.device)
        mention_rep = mention_rep.view(mention_rep.shape[0]*mention_rep.shape[1], -1)

        bag_size = len(sentences[0])

        t0 = time.time()
        # sentences: (batch_size, bag_size, sentence_length [variable])

        # get max sentence length
        max_sent_len = -1
        for batch_idx, item in enumerate(sentences):  # iter through batch
            for sent_idx, sent in enumerate(item):  # iter through bag
                max_sent_len = max(max_sent_len, len(sent))

        sentences_embed = torch.empty((len(sentences) * bag_size, max_sent_len, self.embedding_shape[-1]), device=config.device)

        for batch_idx, item in enumerate(sentences): # iter through batch
            for sent_idx, sent in enumerate(item): # iter through bag
                sent_tensor = torch.tensor(sent, dtype=int, device=config.device)

                sentence_embed = self.embeddings(sent_tensor)

                sentence_pad = F.pad(sentence_embed, pad=(0, 0, 0, max_sent_len - sent_tensor.shape[0]))

                sentences_embed[(batch_idx * bag_size) + sent_idx] = sentence_pad

        t1 = time.time()

        sentences_embed = self.lin1(sentences_embed)

        cnn_out = self.cnn(sentences_embed.permute(0, 2, 1))
        cnn_out, _ = cnn_out.max(dim=-1) # (batch_size * bag_size, embedding_dim)

        mention_rep = self.dropout(mention_rep)

        concat_out = torch.cat((cnn_out, mention_rep), dim=-1).float() # (batch_size * bag_size, embedding_dim * 2)

        concat_out = self.tanh(self.lin2(concat_out))

        out = self.dropout(self.lin3(concat_out))

        if config.timing:
            print('CNNEncoder', time.time() - t1, t1 - t0)

        return out