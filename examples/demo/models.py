from itertools import chain, product
import torch


def tokenize(text):
    words = [text_.split(' ') for text_ in text]
    flatten = list(chain(*words))
    mapping = torch.zeros(len(flatten), len(text))
    i = 0
    for j, words_ in enumerate(words):
        mapping[i:i+len(words_), j] = 1
        i += len(words_)
    return mapping, flatten


class WordEmbedding(torch.nn.Module):
    i2w = ['__UNK__', 'John', 'works', 'for', 'IBM']
    w2i = dict((w,i) for i, w in enumerate(i2w))

    def __init__(self):
        super().__init__()
        self.embeds = torch.nn.Embedding(len(self.i2w), 100)

    def ws2i(self, words):
        ids = [self.w2i.get(word, 0) for word in words]
        device = next(self.embeds.parameters()).device
        return torch.tensor(ids, device=device)

    def forward(self, words):
        ids = self.ws2i(words)
        return self.embeds(ids)


class Classifier(torch.nn.Linear):
    def __init__(self, dim_in=100):
        super().__init__(dim_in, 2)


def make_pair(word_text):
    n = len(word_text)
    arg1i = []
    arg2i = []
    for arg1, arg2 in product(range(n), repeat=2):
        if arg1 == arg2:
            continue
        arg1i.append(arg1)
        arg2i.append(arg2)
    arg1i = torch.tensor(arg1i).unsqueeze(-1)
    arg1m = torch.zeros(arg1i.shape[0], n)
    arg1m.scatter_(1, arg1i, 1)
    arg2i = torch.tensor(arg2i).unsqueeze(-1)
    arg2m = torch.zeros(arg2i.shape[0], n)
    arg2m.scatter_(1, arg2i, 1)

    return arg1m, arg2m


def concat(arg1emb, arg2emb):
    return torch.cat((arg1emb, arg2emb), -1)


def pair_label(arg1m, arg2m, data):
    label = torch.zeros(arg1m.shape[0], dtype=torch.long)
    for arg1, arg2 in data:
        i, = (arg1m[:, arg1] * arg2m[:, arg2]).nonzero(as_tuple=True)
        label[i] = 1
    return label
