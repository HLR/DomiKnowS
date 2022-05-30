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
    def __init__(self, max_length=256, device='cpu'):
        self.max_length = max_length
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.device = device

    def __call__(self, premise, hypothesis):
        encoded_input = self.tokenizer(premise, hypothesis, padding="max_length", max_length=self.max_length)
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]
        return torch.LongTensor(input_ids).to(self.device), torch.LongTensor(attention_mask).to(self.device)


class NLI_model(nn.Module):
    def __init__(self, device ="cpu"):
        super(NLI_model, self).__init__()
        self.tokenizer = NLI_RobertaTokenizer(device=device)
        self.robert = NLI_Robert(device=device)
        self.device = device
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
    def __init__(self, device="cpu"):
        super(NLI_Robert, self).__init__()
        self.bert = RobertaModel.from_pretrained('roberta-base')
        self.last_layer_size = self.bert.config.hidden_size
        self.device = device

    def forward(self, input_ids, attention_mask):
        last_hidden_state, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                                     return_dict=False)
        return last_hidden_state[:, 0].to()


class NLIDataset(Dataset):
    __slots__ = ["data", "target"]

    def __init__(self, file, size, augmented_file=None):
        self.data = []
        self.target = []
        df = pd.read_csv(file).dropna() if file else None
        df_augment = pd.read_json(augmented_file, lines=True).dropna() if augmented_file else None
        all_data = []
        size = min(size, len(df)) if file else None
        if file:
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
            if augmented and data['gold_label'] not in augmented_class:
                data['gold_label'] = 'contradiction'
            target = data['label'] if not augmented else augmented_class[data['gold_label']]
            if target < 0:
                target = 2
            self.data.append((premise, hypo))
            self.target.append(target)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        data = self.data[ind]
        target = self.target[ind]

        return data, torch.tensor(target)


def load_data(file, size, batch_size = 1, *, augment=None):
    return DataLoader(NLIDataset(file=file, size=size, augmented_file=augment), batch_size=batch_size)


def train(model, dataloader, loss_fn, optimizer, epoch, *, device="cpu"):
    model.train()
    losses = []
    for batch, (attr, label) in enumerate(tqdm(dataloader, desc="Training Epoch " + str(epoch))):
        label = torch.LongTensor(label).to(device)
        pred = model(*attr).to(device)
        loss = loss_fn(pred, label)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return np.mean(losses)


def eval(model, dataloader, loss_fn):
    model.eval()
    correct = 0
    total = len(dataloader.dataset)
    with torch.no_grad():
        for attr, label in tqdm(dataloader, desc="Testing"):
            pred = model(*attr)
            correct += (pred.argmax(axis=1) == label).float().sum()
    return correct / total



def main(args):
    train_set = load_data("data/train.csv", size=args.training_sample, batch_size=args.batch_size)
    test_set = load_data("data/test.csv", size=args.testing_sample, batch_size=args.batch_size)
    aug_set = load_data(None, size=0, batch_size=args.batch_size, augment="data/snli_genadv_1000_test.jsonl")
    cuda_number = args.cuda_number
    device = "cuda:" + str(cuda_number) if torch.cuda.is_available() else 'cpu'
    loss_fn = nn.CrossEntropyLoss()
    model = NLI_model(device=device).to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epoch):
        loss = train(model, train_set, loss_fn, optimizer, epoch + 1, device=device)
    accuracy = 100 * eval(model, test_set, loss_fn)
    aug_acc = 100 * eval(model, aug_set, loss_fn)
    print("Accuracy = {:3f}%".format(accuracy))
    print("Augmented Accuracy = {:3f}%".format(aug_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Pytorch NLI")

    parser.add_argument('--cuda', dest='cuda_number', default=0, help='cuda number to train the models on', type=int)

    parser.add_argument('--epoch', dest='epoch', default=10, help='number of epochs to train model', type=int)

    parser.add_argument('--lr', dest='learning_rate', default=1e-6, help='learning rate of the adamW optimiser',
                        type=float)

    parser.add_argument('--training_sample', dest='training_sample', default=550146,
                        help="number of data to train model", type=int)

    parser.add_argument('--testing_sample', dest='testing_sample', default=10000, help="number of data to test model",
                        type=int)

    parser.add_argument('--batch_size', dest='batch_size', default=4, help="batch size of sample", type=int)
    args = parser.parse_args()
    main(args)
