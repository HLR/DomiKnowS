import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn import Flatten


DATA_PATH = 'data'


def get_readers():
    transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              Flatten(0)
                              ])
    trainset = datasets.MNIST(DATA_PATH, download=True, train=True, transform=transform)
    testset = datasets.MNIST(DATA_PATH, download=True, train=False, transform=transform)
    trainloader = DataLoader(trainset,# batch_size=1,
        # sampler=random.sample(range(60000),600),
        shuffle=True,
        )
    testloader = DataLoader(testset,# batch_size=1,
        # sampler=random.sample(range(10000),100),
        shuffle=True,
        )

    return trainloader, testloader
