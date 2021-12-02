from torch import nn
import torch
import time
class POSLSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size,vocabulary,device):
        super(POSLSTM, self).__init__()
        self.device=device
        self.vocabulary=vocabulary
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

    def forward(self, sentence):
        #print("size of the sentefcne:",len(sentence))

        t=time.time()
        vectorizer=[self.vocabulary["<begin>"]]
        for i in sentence:
            vectorizer.append(self.vocabulary[i])
        vectorizer.append(self.vocabulary["<end>"])
        #print("vectorizer",time.time()-t)
        embeds = self.word_embeddings(torch.LongTensor(vectorizer).to(self.device).unsqueeze(0))
        #print("embed",time.time() - t)
        #print(embeds.shape)
        lstm_out, _ = self.lstm(embeds)
        #print("lstm",time.time() - t)
        #print(lstm_out.shape)
        return lstm_out[0][1:-1]

class HeadLayer(nn.Module):

    def __init__(self, hidden_dim, target_size):
        super(HeadLayer, self).__init__()
        self.linear_layer = nn.Linear(hidden_dim, target_size)

    def forward(self, x):
        #t=time.time()
        out = self.linear_layer(x)
        #print("linear time: ",time.time()-t)
        return out