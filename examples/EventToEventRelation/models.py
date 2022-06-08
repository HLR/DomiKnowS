import torch
from transformers import RobertaTokenizer, RobertaModel
from utils import *
from torch import nn


class Roberta_Tokenizer:
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', unk_token='<unk>')

    def __call__(self, content, token_list=None, token_span_SENT=None):
        encoded = self.tokenizer.encode(content)
        roberta_subword_to_ID = encoded
        roberta_subwords = []
        roberta_subwords_no_space = []
        for index, i in enumerate(encoded):
            r_token = self.tokenizer.decode([i])
            roberta_subwords.append(r_token)
            if r_token[0] == " ":
                roberta_subwords_no_space.append(r_token[1:])
            else:
                roberta_subwords_no_space.append(r_token)

        roberta_subword_span = tokenized_to_origin_span(content, roberta_subwords_no_space[1:-1])  # w/o <s> and </s>
        roberta_subword_map = []
        if token_span_SENT is not None:
            roberta_subword_map.append(-1)  # "<s>"
            for subword in roberta_subword_span:
                roberta_subword_map.append(token_id_lookup(token_span_SENT, subword[0], subword[1]))
            roberta_subword_map.append(-1)  # "</s>"
            return roberta_subword_to_ID, roberta_subwords, roberta_subword_span, roberta_subword_map
        else:
            return roberta_subword_to_ID, roberta_subwords, roberta_subword_span, -1


class RobertaToken:
    def __init__(self, cuda="cpu"):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', unk_token='<unk>')

    def __call__(self, _, content):
        encodeds_input = self.tokenizer(content)
        input_id = encodeds_input["input_ids"]
        return [torch.LongTensor(padding(encode)) for encode in input_id]


class RobertaLSTM(nn.Module):
    def __init__(self, roberta_size):
        super(RobertaLSTM, self).__init__()
        self.roberta_model = RobertaModel.from_pretrained(roberta_size)
        self.roberta_last_size = 768 if roberta_size == 'roberta-base' else 1024

    def forward(self, sents):
        return_list = []
        for sent in sents:
            return_list.append(self.roberta_model(sent.unsqueeze(0))[0].view(-1, self.roberta_last_size))
        return torch.stack(return_list)


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, roberta_size, cuda):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=True)
        self.last_layer_size = hidden_size * 2  # Bi direction
        self.cuda = cuda
        self.roberta_model = RobertaLSTM(roberta_size)

    def forward(self, input_sent, pos):
        last_hidden_state, _ = self.lstm(self.roberta_model(input_sent))  # Size [batch_size, 78, 256]
        return torch.flatten(last_hidden_state[0, pos.long(), :].unsqueeze(0), start_dim=0, end_dim=1)


class BiLSTM_MLP(nn.Module):
    def __init__(self, BiLSTM_last_hidden_size, MLP_size, output_classes):
        super(BiLSTM_MLP, self).__init__()
        self.start = nn.Linear(BiLSTM_last_hidden_size * 4, MLP_size * 2)
        self.final = nn.Linear(MLP_size * 2, output_classes)

    def forward(self, x):
        input = x
        x = self.start(input)
        x = torch.relu(x)
        x = self.final(x)
        return x


class Robert_Model(nn.Module):
    def __init__(self):
        super(Robert_Model, self).__init__()
        self.model = RobertaModel.from_pretrained('roberta-base')
        self.last_layer_size = self.model.config.hidden_size

    def forward(self, input_sent, pos):
        last_hidden_state = self.model(input_sent)[0]
        return torch.flatten(last_hidden_state[0, pos.long(), :].unsqueeze(0), start_dim=0, end_dim=1)
