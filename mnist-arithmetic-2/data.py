import random
random.seed(10)
#random.seed(12)

import torch
torch.manual_seed(10)

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.nn import Flatten

import config

DATA_PATH = 'data'


def distr_string(lst):
    return ' || '.join(['%d: %d' % (i, n) for i, n in enumerate(lst)])


class SumBalanceDataset:
    def __init__(self, dataset, digit_ids, length, post_epoch_shuffle=True, verbose=True, eval=False):
        digit_to_id = {}

        for d in range(10):
            digit_to_id[d] = []

        for d_id in digit_ids:
            digit_to_id[dataset[d_id][1]].append(d_id)

        self.num_train = length
        self.digit_to_id = digit_to_id
        self.dataset = dataset

        self.eval = eval

        self.post_epoch_shuffle = post_epoch_shuffle

        self.verbose = verbose

        self.digit_id_pairs = self.build_balanced_sum()

    def build_balanced_sum(self):
        digit_id_pairs = []

        sum_distr = [0] * 19
        digit0_distr = [0] * 10
        digit1_distr = [0] * 10

        while len(digit_id_pairs) < len(self):
            s = random.randint(0, 18)
            d0, d1 = self.sample_digits_from_sum(s)

            sum_distr[s] += 1
            digit0_distr[d0] += 1
            digit1_distr[d1] += 1

            d0_id = random.sample(self.digit_to_id[d0], 1)[0]
            d1_id = random.sample(self.digit_to_id[d1], 1)[0]

            if d0_id == d1_id:
                continue

            self.digit_to_id[d0].remove(d0_id)
            self.digit_to_id[d1].remove(d1_id)

            digit_id_pairs.append((d0_id, d1_id, d0, d1))

        if self.verbose:
            print('SUM DISTRIBUTION')
            print(distr_string(sum_distr))
            print('DIGIT 0 DISTRIBUTION')
            print(distr_string(digit0_distr))
            print('DIGIT 1 DISTRIBUTION')
            print(distr_string(digit1_distr))

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

        if idx == len(self) - 1 and self.post_epoch_shuffle:
            random.shuffle(self.digit_id_pairs)

            if self.verbose:
                print('SumBalanceDataset: Shuffling dataset')

        return {
            'pixels': torch.stack((d0_image[0], d1_image[0]), dim=0),
            'summation': torch.tensor([[d0 + d1]]),
            'digit': [d0, d1],
            'eval': self.eval
        }


def make_sum(samples, eval=False):
    return {
        'pixels': torch.stack(tuple(map(lambda s: s[0], samples)), dim=0),
        'summation': torch.tensor([[sum(map(lambda s: s[1], samples))]]),
        'digit': [samples[0][1], samples[1][1]],
        'eval': eval
    }


def get_readers(num_train):
    transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              Flatten(0)
                              ])
    trainset = datasets.MNIST(DATA_PATH, download=True, train=True, transform=transform)
    testset = datasets.MNIST(DATA_PATH, download=True, train=False, transform=transform)

    assert num_train * 4 <= 50000 and config.num_valid * 2 <= 10000 and config.num_test * 2 <= 10000

    # need to sample twice as many train ids because we're sampling s.t. the sum distribution is uniform
    # so some digits may appear disproportionately often
    # sample pool needs to consist of a minimum of 500 examples as there needs to be enough
    # digits for each combination sampled
    train_ids = random.sample(range(0, 50000), max(num_train * 4, 500))
    valid_ids = random.sample(range(50000, 60000), config.num_valid * 2)
    test_ids = random.sample(range(10000), config.num_test * 2)

    train_balanced = SumBalanceDataset(trainset, train_ids, num_train, verbose=False)

    trainloader = train_balanced

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
        collate_fn=lambda x: make_sum(x, eval=True)
        )
    testloader = DataLoader(
        testset,
        sampler=test_ids,
        shuffle=False,
        batch_size=2,
        collate_fn=lambda x: make_sum(x, eval=True)
        )

    return trainloader, trainloader_mini, validloader, testloader
