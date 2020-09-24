import torch


def tokenize(text):
    return text.split(' ')


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
    def __init__(self):
        super().__init__(100, 2)
