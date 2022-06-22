from typing import Any
import torch

from regr.sensor.pytorch.sensors import FunctionalSensor, JointSensor
from regr.sensor.pytorch.relation_sensors import EdgeSensor


class Tokenizer(EdgeSensor, JointSensor):
    def __init__(self, *args, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        if not tokenizer:
            raise ValueError('You should select a default Tokenizer')
        self.tokenizer = tokenizer

    def forward(self, text) -> Any:
        tokenized = self.tokenizer.encode_plus(text, return_tensors="pt")
        tokens = tokenized['input_ids'].view(-1)
        lens = tokenized['attention_mask'].sum(-1)

        mapping = torch.zeros(tokens.shape[0], lens.shape[0])
        i = 0
        for j, len_ in enumerate(lens):
            mapping[i:i+len_, j] = 1
            i += len_

        return mapping, tokens


class TokenizerSpan(FunctionalSensor):
    def __init__(self, *args, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        if not tokenizer:
            raise ValueError('You should select a default Tokenizer')
        self.tokenizer = tokenizer

    def forward(self, text) -> Any:
        ids_index = []
        start = 0
        for item in self.tokenizer.convert_ids_to_tokens(text, skip_special_tokens=True):
            if item[0] == "Ġ":
                ids_index.append((start + 1, start + len(item) - 1))
            else:
                ids_index.append((start, start + len(item) - 1))
            start += len(item)
        return [(None, None)] + ids_index + [(None, None)]
