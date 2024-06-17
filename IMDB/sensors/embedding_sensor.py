import sys
sys.path.append("../../../")

from domiknows.sensor.pytorch.sensors import FunctionalSensor
from torchtext.vocab import GloVe
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
import torch

class EmbeddingSensor(FunctionalSensor):
  def __init__(self, *pres, embed_size=300, **kwarg):
    super().__init__(*pres, **kwarg)

    self.embed_size = embed_size

    # init GloVe embeddings and tokenizer
    self.embedding = GloVe(name='840B', dim=self.embed_size)
    self.tokenizer = get_tokenizer('spacy', language='en')

  def forward(self, *inputs):
    text = inputs[0]

    # tokenize sentence
    tokens_batch = [self.tokenizer(text)]

    # convert to GloVe embedding vectors
    emb_batch = []
    for tokens in tokens_batch:
      rev_emb = torch.empty((len(tokens), self.embed_size))
      for i, tok in enumerate(tokens):
        rev_emb[i] = self.embedding[tok]
      
      emb_batch.append(rev_emb)

    padded = pad_sequence(emb_batch)

    out = padded.to(device=self.device)

    return out