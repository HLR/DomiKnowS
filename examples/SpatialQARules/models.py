from transformers import BertModel, BertPreTrainedModel, BertTokenizer
from torch import nn
import torch
from torch.autograd import Variable


class BERTTokenizer:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __call__(self, _, question, story):
        encoded_input = self.tokenizer(question, story, padding="max_length", truncation=True)
        input_ids = encoded_input["input_ids"]
        return torch.LongTensor(input_ids)


class MultipleClassYN(BertPreTrainedModel):
    def __init__(self, config, device="cpu", drp=False):
        super().__init__(config)

        if drp:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        self.cur_device = device
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_classes = 3
        self.classifier = nn.Linear(config.hidden_size, self.num_classes)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, input_ids):
        outputs = self.bert(input_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        output = self.classifier(pooled_output)

        return self.softmax(output)