import random
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn import Flatten


def make_sum(samples):
    return {
        'pixels': torch.cat(map(lambda s: s[0], samples), dim=0),
        'summation': sum(map(lambda s: s[1], samples)),
    }

def get_readers():
    transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              Flatten(0)
                              ])
    trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
    testset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
    trainloader = DataLoader(
        trainset,
        # sampler=random.sample(range(60000),600),
        shuffle=True,
        batch_size=2,
        collate_fn=make_sum
        )
    testloader = DataLoader(
        testset,
        # sampler=random.sample(range(10000),100),
        shuffle=True,
        batch_size=2,
        collate_fn=make_sum
        )

    return trainloader, testloader
