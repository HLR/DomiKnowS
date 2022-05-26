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
import torchvision.transforms as transforms


class NLI_RobertaTokenizer:
    def __init__(self, max_length=256):
        self.max_length = max_length
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    def __call__(self, premise, hypothesis):
        encoded_input = self.tokenizer(premise, hypothesis, padding="max_length", max_length=self.max_length)
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]
        return torch.LongTensor(input_ids), torch.LongTensor(attention_mask)


class NLI_model(nn.Module):
    def __init__(self):
        super(NLI_model, self).__init__()
        self.tokenizer = NLI_RobertaTokenizer()
        self.robert = NLI_Robert()
        self.MLP = nn.Sequential(
            nn.Linear(self.robert.last_layer_size, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 3)
        )

    def forward(self, premise, hypothesis):
        input_id, mask = self.tokenizer(premise, hypothesis)
        last_hidden_state = self.robert(input_id, mask)
        return self.MLP(last_hidden_state)


class NLI_Robert(nn.Module):
    def __init__(self):
        super(NLI_Robert, self).__init__()
        self.bert = RobertaModel.from_pretrained('roberta-base')
        self.last_layer_size = self.bert.config.hidden_size

    def forward(self, input_ids, attention_mask):
        last_hidden_state, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                                     return_dict=False)
        return last_hidden_state[:, 0]


class NLIDataset(Dataset):
    __slots__ = ["data", "target"]

    def __init__(self, file, size, model=None, augmented_file=None):
        self.data = []
        self.target = []
        self.model = model
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

        if self.model:
            data = self.model(*data)

        return data, torch.tensor(target)


def load_data(file):
    return DataLoader(NLIDataset(file=file, size=100), batch_size=10)


def main(args):
    test = load_data("data/test.csv")
    cuda_number = args.cuda_number
    device = "cuda:" + str(cuda_number) if torch.cuda.is_available() else 'cpu'
    loss_fn = nn.CrossEntropyLoss()
    model = NLI_model().to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Pytorch NLI")

    parser.add_argument('--cuda', dest='cuda_number', default=0, help='cuda number to train the models on', type=int)

    parser.add_argument('--epoch', dest='cur_epoch', default=10, help='number of epochs to train model', type=int)

    parser.add_argument('--lr', dest='learning_rate', default=1e-6, help='learning rate of the adamW optimiser',
                        type=float)

    parser.add_argument('--training_sample', dest='training_sample', default=550146,
                        help="number of data to train model", type=int)

    parser.add_argument('--testing_sample', dest='testing_sample', default=10000, help="number of data to test model",
                        type=int)

    parser.add_argument('--batch_size', dest='batch_size', default=4, help="batch size of sample", type=int)
    args = parser.parse_args()
    main(args)
