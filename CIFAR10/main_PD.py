import sys
import torch
sys.path.append('.')
sys.path.append('../..')

from domiknows.program import SolverPOIProgram, POIProgram, IMLProgram
from domiknows.program.model.pytorch import PoiModel, IMLModel, SolverModel

from domiknows.program.model.lossModel import PrimalDualModel
from domiknows.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
import os,pickle
import numpy as np
from domiknows.program.loss import NBCrossEntropyLoss, BCEWithLogitsIMLoss
from torch.utils.data import random_split
from domiknows.program.lossprogram import PrimalDualProgram


class ImageNetwork(torch.nn.Module):
    def __init__(self):
        super(ImageNetwork, self).__init__()
        """CNN Builder."""
        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.size(0), -1)

        return x

class LinearNetwork(torch.nn.Module):
    def __init__(self):
        super(LinearNetwork, self).__init__()

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        """Perform forward."""
        # fc layer
        x = self.fc_layer(x)
        return x
        
from graph import graph, image, truck, dog, airplane, automobile, bird, cat, deer, frog, horse, ship

def model_declaration():
    from domiknows.sensor.pytorch.sensors import ReaderSensor
    from domiknows.sensor.pytorch.learners import ModuleLearner
    from domiknows.program import LearningBasedProgram
    from torch import nn
    graph.detach()
    image = graph['image']
    airplane = graph['airplane']
    dog = graph['dog']
    truck = graph['truck']
    automobile = graph['automobile']
    bird = graph['bird']
    cat = graph['cat']
    deer = graph['deer']
    frog = graph['frog']
    horse = graph['horse']
    ship = graph['ship']

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    image['pixels'] = ReaderSensor(keyword='pixels', device=device)
    image[airplane] = ReaderSensor(keyword='airplane',label=True, device=device)
    image[dog] = ReaderSensor(keyword='dog',label=True, device=device)
    image[truck] = ReaderSensor(keyword='truck',label=True, device=device)
    image[automobile] = ReaderSensor(keyword='automobile',label=True, device=device)
    image[bird] = ReaderSensor(keyword='bird',label=True, device=device)
    image[cat] = ReaderSensor(keyword='cat',label=True, device=device)
    image[deer] = ReaderSensor(keyword='deer',label=True, device=device)
    image[frog] = ReaderSensor(keyword='frog',label=True, device=device)
    image[horse] = ReaderSensor(keyword='horse',label=True, device=device)
    image[ship] = ReaderSensor(keyword='ship',label=True, device=device)

    image['emb'] = ModuleLearner('pixels', module=ImageNetwork(), device=device)
    image[airplane] = ModuleLearner('emb', module=LinearNetwork(), device=device)
    image[dog] = ModuleLearner('emb', module=LinearNetwork(), device=device)
    image[truck] = ModuleLearner('emb', module=LinearNetwork(), device=device)
    image[automobile] = ModuleLearner('emb', module=LinearNetwork(), device=device)
    image[bird] = ModuleLearner('emb', module=LinearNetwork(), device=device)
    image[cat] = ModuleLearner('emb', module=LinearNetwork(), device=device)
    image[deer] = ModuleLearner('emb', module=LinearNetwork(), device=device)
    image[frog] = ModuleLearner('emb', module=LinearNetwork(), device=device)
    image[horse] = ModuleLearner('emb', module=LinearNetwork(), device=device)
    image[ship] = ModuleLearner('emb', module=LinearNetwork(), device=device)

    program = PrimalDualProgram(graph, SolverModel, poi=(image, ), inferTypes=['ILP', 'local/argmax'], loss=MacroAverageTracker(NBCrossEntropyLoss()), metric={'ILP':PRF1Tracker(DatanodeCMMetric()),'softmax':PRF1Tracker(DatanodeCMMetric('local/argmax'))})

    return program

class CIFAR10_1(datasets.CIFAR10):

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):

        super(CIFAR10_1, self).__init__(root, transform=transform,
                                      target_transform=target_transform, download=download)

        self.train = train  # training set or test set

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')

                self.data.append(entry['data'])

                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        img = img.unsqueeze(0)
        target_dict = {0:'airplane',1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7:'horse',8: 'ship', 9: 'truck'}
        dict = {}
        dict['pixels'] = img
        for i in range(10):
            dict[target_dict[i]] = [0]
        dict[target_dict[target]] = [1]
        return dict

def load_cifar10(train=True, root='./data/', size=32):
    CIFAR100_TRAIN_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR100_TRAIN_STD = (0.2023, 0.1994, 0.2010)

    if train:
        transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
             transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)])
    else:
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)])

    return CIFAR10_1(root=root, train=train, transform=transform,download=True)

def main():

    program = model_declaration()

    ### load data
    val_size = 5000

    trainset = load_cifar10(train=True)
    testset = load_cifar10(train=False)
    train_size = len(trainset) - val_size
    train_ds, val_ds = random_split(trainset, [train_size, val_size])
    print(len(train_ds), len(val_ds))

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    program.train(training_set=train_ds, valid_set=val_ds, test_set=testset, device=device, train_epoch_num=50, Optim=lambda param: torch.optim.SGD(param, lr=.001))
    
    program.save("./cifar_PD")
    
    program.test(testset)
        

if __name__ == '__main__':
    main()
