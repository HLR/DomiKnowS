import random
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn import Flatten
from graph_u import digitRange, summationVal

DATA_PATH = 'data'


class DataPair():
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        candidates = dict(map(lambda d: (d, []), range(digitRange)))
        for iPixel, iVal in self.loader:
            jVal = summationVal - iVal
            # print(iVal)
            if candidates[int(jVal)]:
                jPixel, jVal = candidates[int(jVal)].pop()
                # print('pop', iVal, jVal)
                # print(dict((digit, len(candidate)) for digit, candidate in candidates.items()))
                yield {
                    "pixels": torch.cat((iPixel, jPixel), dim=0),
                    "vals": [iVal, jVal], #torch.cat((iVal, jVal), dim=0),
                }
            else:
                candidates[int(iVal)].append((iPixel, iVal))
                # print('push', iVal)
                # print(dict((digit, len(candidate)) for digit, candidate in candidates.items()))
                continue

    def __len__(self):
        return len(self.loader) // 2

def get_readers():
    transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              Flatten(0)
                              ])
    trainset = datasets.MNIST(DATA_PATH, download=True, train=True, transform=transform)
    testset = datasets.MNIST(DATA_PATH, download=True, train=False, transform=transform)
    train_single_loader = DataLoader(
        trainset,
        sampler=random.sample(range(60000),600),
        # shuffle=True,
        batch_size=1,
        )
    train_pair_loader = DataPair(train_single_loader)
    test_single_loader = DataLoader(
        testset,
        sampler=random.sample(range(10000),100),
        # shuffle=False,
        batch_size=1,
        )
    test_pair_loader = DataPair(test_single_loader)

    return train_pair_loader, test_pair_loader
