import torch
from regr.sensor.pytorch.learners import TorchLearner
from typing import Any


class TransformerRepresentations(TorchLearner):
    def __init__(self, *pre, edges=None, loss=None, metric=None, label=False, device='auto', transformer=None):
        super(TransformerRepresentations, self).__init__(*pre, edges, loss, metric, label, device)
        if not transformer:
            raise ValueError('You should select a default Transformer')
        self.model = transformer

    def forward(self, ) -> Any:
        outputs = self.model(input_ids=self.inputs[0].unsqueeze(0),
                             attention_mask=torch.ones(self.inputs[0].shape).unsqueeze(0))
        return outputs[0]
