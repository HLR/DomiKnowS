from regr.sensor.pytorch.sensors import TorchSensor, TorchEdgeSensor, TriggerPrefilledSensor
from typing import Any


class Tokenizer(TorchEdgeSensor):
    def __init__(self, *pres, to, mode="forward", edges=None, label=False, device='auto', tokenizer=None):
        super().__init__(*pres, to=to, mode=mode, edges=edges, label=label, device=device)
        if not tokenizer:
            raise ValueError('You should select a default Tokenizer')
        self.tokenizer = tokenizer

    def forward(self, text) -> Any:
        tokenized = self.tokenizer.encode_plus(text, return_tensors="pt", return_offsets_mapping=True)
        tokens = tokenized['input_ids'].view(-1).to(device=self.device)
        offset = tokenized['offset_mapping'].to(device=self.device)
        return tokens, offset

    def attached(self, sup):
        super(TorchEdgeSensor, self).attached(sup)  # skip TorchEdgeSensor
        self.relation = sup.sup
        if self.mode == "forward":
            self.src = self.relation.src
            self.dst = self.relation.dst
        elif self.mode == "backward" or self.mode == "selection":
            self.src = self.relation.dst
            self.dst = self.relation.src
        else:
            raise ValueError('The mode passed to the edge is invalid!')
        if isinstance(self.to, tuple):
            for to_ in self.to:
                self.dst[to_] = TriggerPrefilledSensor(callback_sensor=self)
        else:
            self.dst[self.to] = TriggerPrefilledSensor(callback_sensor=self)

    def update_context(self, data_item, force=False):
        super(TorchEdgeSensor, self).update_context(data_item, force=force)  # skip TorchEdgeSensor
        if isinstance(self.to, tuple):
            for to_, value in zip(self.to, data_item[self.fullname]):
                data_item[self.dst[to_].fullname] = value
        else:
            data_item[self.dst[self.to].fullname] = data_item[self.fullname]
        return data_item

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
