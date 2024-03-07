import torch
from torch import nn
from transformers import T5ForConditionalGeneration

class FilteredT5Model(nn.Module):
    def __init__(self, model_name, tokenizer):
        super(FilteredT5Model, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(tokenizer))

    def forward(self,_, inputs, tags):
        outputs = self.model(input_ids=inputs,decoder_input_ids=torch.cat([torch.full((1,), self.model.config.pad_token_id, dtype=torch.long).to(tags.device), tags.view(-1)[ :-1]], dim=0).unsqueeze(0)) 
        return outputs.logits[0,:,-3:]
