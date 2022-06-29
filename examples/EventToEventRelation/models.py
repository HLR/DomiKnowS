import torch
from transformers import RobertaTokenizer, RobertaModel
from utils import *
from torch import nn
import torch.nn.functional as F
import numpy as np


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
    def __init__(self, max_length=512):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', unk_token='<unk>')
        self.max_length = max_length

    def __call__(self, content):
        encoded_input = self.tokenizer(content, padding="max_length", max_length=self.max_length)
        input_id = encoded_input["input_ids"]
        return torch.LongTensor(input_id)


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
    def __init__(self, input_size, hidden_size, num_layers, roberta_size):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=True)
        self.last_layer_size = hidden_size * 2  # Bi direction
        self.roberta_model = RobertaLSTM(roberta_size)

    def forward(self, input_sent, pos):
        last_hidden_state, _ = self.lstm(self.roberta_model(input_sent))  # Size [batch_size, 78, 256]
        return torch.flatten(last_hidden_state[0, pos.long(), :].unsqueeze(0), start_dim=0, end_dim=1)


class BiLSTM_MLP(nn.Module):
    def __init__(self, BiLSTM_last_hidden_size, MLP_size, output_classes):
        super(BiLSTM_MLP, self).__init__()
        self.start = nn.Linear(BiLSTM_last_hidden_size * 5, MLP_size * 2)
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


class common_sense_from_NN:
    def __init__(self, pre_emb, pre_NN, ratio=0.3, layer=1, emb_size=512, device="cpu"):
        self.verb_map = {}
        verb_emb_file = open(pre_emb)
        lines = verb_emb_file.readlines()
        for ind, line in enumerate(lines):
            verb = line.split()[0]
            self.verb_map[verb] = ind
        verb_emb_file.close()
        self.model = VerbNN(len(self.verb_map), ratio=ratio, emb_size=emb_size, layer=layer)
        pre_train = torch.load(pre_NN)
        self.model.load_state_dict(pre_train['model_state_dict'])
        self.cur_device = device

    def eval(self, verb1, verb2):
        return self.model(torch.from_numpy(np.array([[self.verb_map[verb1], self.verb_map[verb2]]])).to(self.cur_device))

    def getCommonSense(self, verb1, verb2):
        if verb1 not in self.verb_map or verb2 not in self.verb_map:
            return torch.FloatTensor([0, 0]).view(1, -1)
        return torch.FloatTensor([0, 0]).view(1, -1).view(1, -1)


class VerbNN(nn.Module):
    def __init__(self, vocab_size, ratio=0.5, emb_size=512, layer=1):
        super(VerbNN, self).__init__()
        self.emb_size = emb_size * 2  # Bi-direction
        self.emb_layer = nn.Embedding(vocab_size, self.emb_size)
        self.linear1 = nn.Linear(self.emb_size, int(self.emb_size * ratio))
        self.linear2 = nn.Linear(int(self.emb_size * ratio), 1)

    def forward(self, verb):
        x_emb = self.emb_layer(verb)
        fullX = torch.cat((x_emb[:, 0, :], x_emb[:, 1, :]), dim=1)
        layer1 = F.relu(self.linear1(F.dropout(fullX, p=0.3, training=True)))
        return torch.sigmoid(self.fc2(layer1))
