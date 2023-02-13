import sys
import torch
sys.path.append('.')
sys.path.append('../..')

from regr.program import SolverPOIProgram, POIProgram, IMLProgram
from regr.program.model.pytorch import PoiModel, IMLModel
from regr.program.model.primaldual import PrimalDualModel
from regr.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
import os,pickle
import numpy as np
from regr.program.loss import NBCrossEntropyLoss, BCEWithLogitsIMLoss
from torch.utils.data import random_split




def prediction_softmax(pr, gt):
    return torch.softmax(pr.data, dim=-1)


class ImageNetwork(torch.nn.Module):
    def __init__(self, n_outputs=2):
        super(ImageNetwork, self).__init__()
        self.num_outputs = n_outputs
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)


        # self.fc = nn.Linear(16 * 5 * 5, n_outputs)

        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc = nn.Linear(84, n_outputs)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16 * 5 * 5)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc(x)
        return x

        
from graph import graph, image, truck, dog, airplane, automobile, bird, cat, deer, frog, horse, ship

def model_declaration():
    from regr.sensor.pytorch.sensors import ReaderSensor
    from regr.sensor.pytorch.learners import ModuleLearner
    from regr.program import LearningBasedProgram
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

    image['pixels'] = ReaderSensor(keyword='pixels', device="cuda:1")
    image[airplane] = ReaderSensor(keyword='airplane',label=True, device="cuda:1")
    image[dog] = ReaderSensor(keyword='dog',label=True, device="cuda:1")
    image[truck] = ReaderSensor(keyword='truck',label=True, device="cuda:1")
    image[automobile] = ReaderSensor(keyword='automobile',label=True, device="cuda:1")
    image[bird] = ReaderSensor(keyword='bird',label=True, device="cuda:1")
    image[cat] = ReaderSensor(keyword='cat',label=True, device="cuda:1")
    image[deer] = ReaderSensor(keyword='deer',label=True, device="cuda:1")
    image[frog] = ReaderSensor(keyword='frog',label=True, device="cuda:1")
    image[horse] = ReaderSensor(keyword='horse',label=True, device="cuda:1")
    image[ship] = ReaderSensor(keyword='ship',label=True, device="cuda:1")

    image['emb'] = ModuleLearner('pixels', module=ImageNetwork(), device="cuda:1")
    image[airplane] = ModuleLearner('emb', module=nn.Linear(16 * 5 * 5, 2), device="cuda:1")
    image[dog] = ModuleLearner('emb', module=nn.Linear(16 * 5 * 5, 2), device="cuda:1")
    image[truck] = ModuleLearner('emb', module=nn.Linear(16 * 5 * 5, 2), device="cuda:1")
    image[automobile] = ModuleLearner('emb', module=nn.Linear(16 * 5 * 5, 2), device="cuda:1")
    image[bird] = ModuleLearner('emb', module=nn.Linear(16 * 5 * 5, 2), device="cuda:1")
    image[cat] = ModuleLearner('emb', module=nn.Linear(16 * 5 * 5, 2), device="cuda:1")
    image[deer] = ModuleLearner('emb', module=nn.Linear(16 * 5 * 5, 2), device="cuda:1")
    image[frog] = ModuleLearner('emb', module=nn.Linear(16 * 5 * 5, 2), device="cuda:1")
    image[horse] = ModuleLearner('emb', module=nn.Linear(16 * 5 * 5, 2), device="cuda:1")
    image[ship] = ModuleLearner('emb', module=nn.Linear(16 * 5 * 5, 2), device="cuda:1")
    


#     program = IMLProgram(graph, poi=(image, ), inferTypes=['ILP', 'softmax'], loss=MacroAverageTracker(BCEWithLogitsIMLoss(lmbd=0.5)), metric={'ILP':PRF1Tracker(DatanodeCMMetric()),'softmax':PRF1Tracker(DatanodeCMMetric('softmax'))})
    program = IMLProgram(graph, poi=(image, ), inferTypes=['ILP', 'local/argmax'], loss=MacroAverageTracker(BCEWithLogitsIMLoss(lmbd=0.5)), metric={'ILP':PRF1Tracker(DatanodeCMMetric()),'softmax':PRF1Tracker(DatanodeCMMetric('local/argmax'))})

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

#                 self.data.append(entry['data'][:100])
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
    CIFAR100_TRAIN_MEAN = (0.5071, 0.4867, 0.4408)
    CIFAR100_TRAIN_STD = (0.2675, 0.2565, 0.2761)

    if train:
        transform = transforms.Compose(
            [transforms.Resize(size),
             transforms.RandomCrop(size, padding=round(size/8)),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)])
    else:
        transform = transforms.Compose(
            [transforms.Resize(size),
             transforms.ToTensor(),
             transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)])

    return CIFAR10_1(root=root, train=train, transform=transform,download=True)

def main():
    
    program = model_declaration()

    ### load data
#     val_size = 50
    val_size = 5000

    trainset = load_cifar10(train=True)
    testset = load_cifar10(train=False)
    train_size = len(trainset) - val_size
    train_ds, val_ds = random_split(trainset, [train_size, val_size])
    print(len(train_ds), len(val_ds))
    
    program.train(training_set=train_ds, valid_set=val_ds, test_set=testset, device="cuda:1", train_epoch_num=30, Optim=lambda param: torch.optim.SGD(param, lr=.001))
    
    program.test(testset)



        

if __name__ == '__main__':
    main()
