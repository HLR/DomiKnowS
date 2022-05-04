import random
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn import Flatten

import config

DATA_PATH = 'data'


def make_sum(samples):
    return {
        'pixels': torch.unsqueeze(torch.stack(tuple(map(lambda s: s[0], samples)), dim=0), dim=0),
        'summation': [sum(map(lambda s: s[1], samples))],
    }


def get_readers():
    transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              Flatten(0)
                              ])
    trainset = datasets.MNIST(DATA_PATH, download=True, train=True, transform=transform)
    testset = datasets.MNIST(DATA_PATH, download=True, train=False, transform=transform)

    assert config.num_train * 2 <= 50000 and config.num_valid * 2 <= 10000 and config.num_test * 2 <= 10000

    trainloader = DataLoader(
        trainset,
        sampler=random.sample(range(0, 2000), config.num_train * 2),
        shuffle=False,
        batch_size=2,
        collate_fn=make_sum
        )
    validloader = DataLoader(
        trainset,
        sampler=random.sample(range(50000), config.num_valid * 2),
        shuffle=False,
        batch_size=2,
        collate_fn=make_sum
        )
    testloader = DataLoader(
        testset,
        sampler=random.sample(range(10000), config.num_test * 2),
        shuffle=False,
        batch_size=2,
        collate_fn=make_sum
        )

    return trainloader, validloader, testloader
