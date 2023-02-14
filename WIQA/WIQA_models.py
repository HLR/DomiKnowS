from transformers import RobertaModel
import torch
from torch import nn
from domiknows.program.model.primaldual import PrimalDualModel

class WIQA_Robert(nn.Module):

    def __init__(self):
        super(WIQA_Robert, self).__init__()
        self.bert = RobertaModel.from_pretrained('roberta-base')
        self.last_layer_size = self.bert.config.hidden_size

    def forward(self, input_ids,attention_mask):
        try:
            last_hidden_state, pooled_output = self.bert(input_ids=input_ids,attention_mask=attention_mask,return_dict=False)
        except RuntimeError:
            print("Cuda out of memory")
            print(torch.cuda.memory_summary(device=None, abbreviated=False))
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            last_hidden_state, pooled_output = self.bert(input_ids=input_ids,attention_mask=attention_mask,return_dict=False)

            return last_hidden_state[:,0]
        return last_hidden_state[:,0]

class RobertaClassificationHead(nn.Module):

    def __init__(self,last_layer_size):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(last_layer_size, last_layer_size)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(last_layer_size, 2)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class RobertaClassificationHeadMultiClass(nn.Module):

    def __init__(self,last_layer_size):
        super(RobertaClassificationHeadMultiClass, self).__init__()
        self.dense = nn.Linear(last_layer_size, last_layer_size)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(last_layer_size, 3)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class WIQAModel(PrimalDualModel):
    def __init__(self, graph, poi, loss, metric):
        super().__init__(
            graph,
            poi=poi,
            loss=loss,
            metric=metric)
