from regr.sensor.pytorch.sensors import TorchSensor, TorchEdgeSensor
from typing import Any


class Tokenizer(TorchEdgeSensor):
    def __init__(self, *pres, to, mode="forward", edges=None, label=False, device='auto', tokenizer=None):
        super().__init__(*pres, to=to, mode=mode, edges=edges, label=label, device=device)
        if not tokenizer:
            raise ValueError('You should select a default Tokenizer')
        self.tokenizer = tokenizer

    def forward(self, text) -> Any:
        tokens = self.tokenizer.encode_plus(text, return_tensors="pt")['input_ids'].view(-1)
        return tokens


class TokenizerSpan(TorchSensor):
    def __init__(self, *pres, edges=None, label=False, device='auto', tokenizer=None):
        super().__init__(*pres, edges=edges, label=label, device=device)
        if not tokenizer:
            raise ValueError('You should select a default Tokenizer')
        self.tokenizer = tokenizer

    def forward(self, ) -> Any:
        ids_index = []
        start = 0
        for item in self.tokenizer.convert_ids_to_tokens(self.inputs[0], skip_special_tokens=True):
            if item[0] == "Ġ":
                ids_index.append((start + 1, start + len(item) - 1))
            else:
                ids_index.append((start, start + len(item) - 1))
            start += len(item)
        print(ids_index)
        return [(None, None)] + ids_index + [(None, None)]
