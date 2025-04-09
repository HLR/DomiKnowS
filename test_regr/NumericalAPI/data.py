import random
random.seed(10)

import torch
torch.manual_seed(10)

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn import Flatten

DATA_PATH = 'data'

def make_sum(samples, do_eval=False):
    return {
        'pixels': torch.stack(tuple(map(lambda s: s[0], samples)), dim=0),
        'summation': torch.tensor([[sum(map(lambda s: s[1], samples))]]),
        'multiplication': torch.tensor([[samples[0][1] * samples[1][1]]]),
        'subtraction': torch.tensor([[samples[0][1] - samples[1][1]]]),
        'division': torch.tensor([[samples[0][1] // samples[1][1]]]) if samples[1][1] != 0 else None,
        'digit': torch.tensor([samples[0][1], samples[1][1]]),
        'eval': torch.tensor(do_eval)
    }

def get_readers(num_train, num_valid = 1000, num_test = 5000):
    transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              Flatten(0)
                              ])
    trainset = datasets.MNIST(DATA_PATH, download=True, train=True, transform=transform)
    testset = datasets.MNIST(DATA_PATH, download=True, train=False, transform=transform)

    assert num_train <= 50000 and num_valid * 2 <= 10000 and num_test * 2 <= 10000

    train_ids = random.sample(range(0, 50000), max(num_train, 500))
    valid_ids = random.sample(range(50000, 60000), num_valid * 2)
    test_ids = random.sample(range(10000), num_test * 2)

    trainloader_mini = DataLoader(
        trainset,
        sampler=train_ids[:num_valid * 2],
        shuffle=False,
        batch_size=2,
        collate_fn=make_sum
    )
    trainloader = DataLoader(
        trainset,
        sampler=train_ids,
        shuffle=False,
        batch_size=2,
        collate_fn=make_sum
    )
    validloader = DataLoader(
        trainset,
        sampler=valid_ids,
        shuffle=False,
        batch_size=2,
        collate_fn=lambda x: make_sum(x, do_eval=True)
    )
    testloader = DataLoader(
        testset,
        sampler=test_ids,
        shuffle=False,
        batch_size=2,
        collate_fn=lambda x: make_sum(x, do_eval=True)
    )

    return trainloader, trainloader_mini, validloader, testloader
