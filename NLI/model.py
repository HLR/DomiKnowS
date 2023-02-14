from transformers import RobertaModel, RobertaTokenizerFast
from torch import nn
from domiknows.sensor.pytorch.sensors import TorchSensor, FunctionalSensor
import spacy
from typing import Any
import torch


class NLI_Robert(nn.Module):
    """
    Roberta model, language model, to create the input for classify layer
    """
    def __init__(self):
        super(NLI_Robert, self).__init__()
        self.bert = RobertaModel.from_pretrained('roberta-base')
        self.last_layer_size = self.bert.config.hidden_size

    def forward(self, input_ids, attention_mask):
        last_hidden_state, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                                     return_dict=False)
        return last_hidden_state[:, 0]


class RobertClassification(nn.Module):
    """
    Classifying layer to predict whether the hypothesis is entailment, contradiction, or neutral
    """
    def __init__(self, last_layer_size, *, hidden_layer_size=1):
        super(RobertClassification, self).__init__()
        self.hidden_size = hidden_layer_size
        self.start = nn.Linear(last_layer_size, 512)
        self.hidden = nn.Linear(512, 512)
        self.final = nn.Linear(512, 3)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.start(x)
        x = torch.relu(x)
        x = self.dropout(x)

        for _ in range(self.hidden_size):
            x = self.hidden(x)
            x = torch.relu(x)
            x = self.dropout(x)

        x = self.final(x)
        return x


class RobertaTokenizerMulti:
    """
    Tokenizer model from RoBERTa
    """
    def __init__(self, max_length=256):
        self.max_length = max_length
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    def __call__(self, _, premise, hypothesis):
        encoded_input = self.tokenizer(premise, hypothesis, padding="max_length", max_length=self.max_length)
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]
        return torch.LongTensor(input_ids), torch.LongTensor(attention_mask)
