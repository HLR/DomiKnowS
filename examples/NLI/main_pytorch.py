import sys

sys.path.append(".")
sys.path.append("../")
sys.path.append("../../")

import pandas as pd
import torch
import argparse
from torch.utils.data import random_split, DataLoader, Dataset
import numpy as np
from torch import nn
from tqdm import tqdm
import torch.optim as optim
from transformers import RobertaModel, RobertaTokenizerFast


class NLI_RobertaTokenizer:
    def __init__(self, max_length=256):
        self.max_length = max_length
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    def __call__(self, hypothesis, premise):
        encoded_input = self.tokenizer(hypothesis, premise, padding="max_length", max_length=self.max_length)
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]
        return torch.LongTensor(input_ids), torch.LongTensor(attention_mask)


class NLI_Robert(nn.Module):

    def __init__(self):
        super(NLI_Robert, self).__init__()
        self.bert = RobertaModel.from_pretrained('roberta-base')
        self.last_layer_size = self.bert.config.hidden_size

    def forward(self, input_ids, attention_mask):
        last_hidden_state, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                                     return_dict=False)
        return last_hidden_state[:, 0]


class NLI_model(nn.Module):

    def __int__(self):
        super(NLI_model, self).__int__()
        self.tokenizer = NLI_RobertaTokenizer()
        self.robert = NLI_Robert()

    def forward(self, x):
        return x


class NLIDataset(Dataset):
    __slots__ = ["data", "target"]

    def __init__(self, file, size, augmented_file=None, transform=None):
        self.data = []
        self.target = []
        self.transform = transform
        df = pd.read_csv(file).dropna()
        df_augment = pd.read_csv(augmented_file) if augmented_file else None
        all_data = []
        sample = df.iloc[:size, :]
        for _, data in sample.iterrows():
            all_data.append((data, False))
        if augmented_file:
            for _, data in df_augment.iterrows():
                all_data.append((data, True))

        augmented_class = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        for data, augmented in all_data:
            premise = data["premise"] if not augmented else data["sentence1"]
            hypo = data["hypothesis"] if not augmented else data["sentence2"]
            target = data['label'] if not augmented else augmented_class[data['gold_label']]
            self.data.append((premise, hypo))
            self.target.append(target)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        data = self.data[ind]
        target = self.target[ind]

        if self.transform:
            data = self.transform(data)

        return data, target
