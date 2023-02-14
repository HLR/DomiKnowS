from domiknows.sensor.pytorch.sensors import JointSensor
from typing import Any
from domiknows.sensor.pytorch.relation_sensors import EdgeSensor


class TokenizerEdgeSensor(EdgeSensor, JointSensor):
    def __init__(self, *pres, relation, edges=None, label=False, device='auto', tokenizer=None):
        super().__init__(*pres, relation=relation, edges=edges, label=label, device=device)
        if not tokenizer:
            raise ValueError('You should select a default Tokenizer')
        self.tokenizer = tokenizer

    def forward(self, text) -> Any:
        tokenized = self.tokenizer.encode_plus(text, return_tensors="pt", return_offsets_mapping=True)
        tokens = tokenized['input_ids'].view(-1).to(device=self.device)
        offset = tokenized['offset_mapping'].to(device=self.device)
        return tokens, offset
