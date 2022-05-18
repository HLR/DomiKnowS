import random
random.seed(10)

import torch
torch.manual_seed(10)

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.nn import Flatten

import config

DATA_PATH = 'data'


class SumBalanceDataset(Dataset):
    def __init__(self, dataset, digit_ids):
        digit_to_id = {}

        for d in range(10):
            digit_to_id[d] = []

        for d_id in digit_ids:
            digit_to_id[dataset[d_id][1]].append(d_id)

        self.num_train = len(digit_ids) // 2
        self.digit_to_id = digit_to_id
        self.dataset = dataset

        self.digit_id_pairs = self.build_balanced_sum()

    def build_balanced_sum(self):
        digit_id_pairs = []

        for i in range(self.num_train):
            s = random.randint(0, 18)
            d0, d1 = self.sample_digits_from_sum(s)

            d0_id = random.sample(self.digit_to_id[d0], 1)[0]
            d1_id = random.sample(self.digit_to_id[d1], 1)[0]

            digit_id_pairs.append((d0_id, d1_id, d0, d1))

        return digit_id_pairs

    def sample_digits_from_sum(self, sum_val):
        d0 = random.randint(max(0, sum_val - 9), min(sum_val, 9))
        d1 = sum_val - d0
        return (d0, d1)

    def __len__(self):
        return self.num_train

    def __getitem__(self, idx):
        d0_id, d1_id, d0, d1 = self.digit_id_pairs[idx]

        d0_image = self.dataset[d0_id]
        d1_image = self.dataset[d1_id]

        return {
            'pixels': torch.unsqueeze(torch.stack((d0_image[0], d1_image[0]), dim=0), dim=0),
            'summation': [d0 + d1],
            'digit0': [d0],
            'digit1': [d1]
        }

def make_sum(samples):
    return {
        'pixels': torch.unsqueeze(torch.stack(tuple(map(lambda s: s[0], samples)), dim=0), dim=0),
        'summation': [sum(map(lambda s: s[1], samples))],
        'digit0': [samples[0][1]],
        'digit1': [samples[1][1]]
    }


def get_readers():
    transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              Flatten(0)
                              ])
    trainset = datasets.MNIST(DATA_PATH, download=True, train=True, transform=transform)
    testset = datasets.MNIST(DATA_PATH, download=True, train=False, transform=transform)

    assert config.num_train * 2 <= 50000 and config.num_valid * 2 <= 10000 and config.num_test * 2 <= 10000

    train_ids = random.sample(range(0, 50000), config.num_train * 2)
    valid_ids = random.sample(range(50000, 60000), config.num_valid * 2)
    test_ids = random.sample(range(10000), config.num_test * 2)

    train_balanced = SumBalanceDataset(trainset, train_ids)

    trainloader = DataLoader(
        train_balanced,
        shuffle=False
    )

    trainloader_mini = DataLoader(
        trainset,
        sampler=train_ids[:config.num_valid * 2],
        shuffle=False,
        batch_size=2,
        collate_fn=make_sum
        )
    '''trainloader = DataLoader(
        trainset,
        sampler=train_ids,
        shuffle=False,
        batch_size=2,
        collate_fn=make_sum
    )'''
    validloader = DataLoader(
        trainset,
        sampler=valid_ids,
        shuffle=False,
        batch_size=2,
        collate_fn=make_sum
        )
    testloader = DataLoader(
        testset,
        sampler=test_ids,
        shuffle=False,
        batch_size=2,
        collate_fn=make_sum
        )

    return trainloader, trainloader_mini, validloader, testloader
