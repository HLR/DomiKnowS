import torch
from typing import Optional


class TokenMap:
    '''
    Maps between indices outputted by the model and indices outputted by the tokenizer.
    '''
    def __init__(self, label_map: dict[int, int], max_length: Optional[int] = None):
        self.label_map = label_map
        self.inv_label_map = {i: label for label, i in label_map.items()}
        
        if max_length is not None:
            inv_map_len = {}
            for i in range(max_length):
                assert i in self.inv_label_map

                inv_map_len[i] = self.inv_label_map[i]
            
            self.inv_label_map = inv_map_len
            self.label_map = {label: i for i, label in inv_map_len.items()}

        print(f'number of vocab items: {len(self.label_map)}')

        self.label_list = [self.inv_label_map[i] for i in range(len(self.inv_label_map))]
    
    def map_vocab(self, tokens: torch.Tensor) -> torch.Tensor:
        assert len(tokens.shape) == 1, "Expected tokens to be 1D, instead got shape: %s" % str(tokens.shape)
        return torch.tensor([self.label_map[int(token)] for token in tokens])

    def unmap_vocab(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.tensor([self.inv_label_map[int(token)] for token in tokens])

    def __len__(self):
        return len(self.label_map)


def tokenize(text, tokenizer):
    out = tokenizer(text, return_tensors='pt').input_ids
    return out

