import torch
from torch import nn
from regr.program.model.primaldual import PrimalDualModel
from transformers import RobertaModel

class Modeling_Roberta(nn.Module):

    def __init__(self):
        super(Modeling_Roberta, self).__init__()
        self.bert = RobertaModel.from_pretrained('roberta-base')

    def forward(self, input_ids,attention_mask):
        hidden_state, _ = self.bert(input_ids=input_ids,attention_mask=attention_mask,return_dict=False)
        return hidden_state[:,0]

class RobertaClassification(nn.Module):

    def __init__(self,last_layer_size, args):
        super(RobertaClassification, self).__init__()
        self.mlp = nn.Linear(last_layer_size, last_layer_size)
        self.dropout = nn.Dropout(args.dropout)
        self.output = nn.Linear(last_layer_size, 2) ## why 2 here?

    def forward(self, x):
        # x = self.dropout(x)
        x = self.mlp(x)
        x = torch.relu(x)
        x = self.dropout(x)
        output = self.output(x)
        return output

class WIQAModel(PrimalDualModel):
    def __init__(self, graph, poi, loss, metric):
        super().__init__(
            graph,
            poi=poi,
            loss=loss,
            metric=metric)
