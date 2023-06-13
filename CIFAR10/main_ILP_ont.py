import sys
import torch
sys.path.append('.')
sys.path.append('../..')

from domiknows.program import SolverPOIProgram, POIProgram, IMLProgram
from domiknows.program.model.pytorch import PoiModel, IMLModel
from domiknows.program.model.lossModel import PrimalDualModel
from domiknows.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from PIL import Image
import torch.nn as nn
import os,pickle
import numpy as np
from domiknows.program.loss import NBCrossEntropyLoss, BCEWithLogitsIMLoss
from torch.utils.data import random_split

# https://zhenye-na.github.io/2018/09/28/pytorch-cnn-cifar10.html
def prediction_softmax(pr, gt):
    return torch.softmax(pr.data, dim=-1)

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
        
from graph_ont import graph_ont, image, truck, dog, airplane, automobile, bird, cat, deer, frog, horse, ship,animal, vehicle

def model_declaration():
    from domiknows.sensor.pytorch.sensors import ReaderSensor
    from domiknows.sensor.pytorch.learners import ModuleLearner
    
    graph_ont.detach()

    image['pixels'] = ReaderSensor(keyword='pixels')
    image[animal] = ReaderSensor(keyword='animal',label=True)
    image[vehicle] = ReaderSensor(keyword='vehicle',label=True)
    image[airplane] = ReaderSensor(keyword='airplane',label=True)
    image[dog] = ReaderSensor(keyword='dog',label=True)
    image[truck] = ReaderSensor(keyword='truck',label=True)
    image[automobile] = ReaderSensor(keyword='automobile',label=True)
    image[bird] = ReaderSensor(keyword='bird',label=True)
    image[cat] = ReaderSensor(keyword='cat',label=True)
    image[deer] = ReaderSensor(keyword='deer',label=True)
    image[frog] = ReaderSensor(keyword='frog',label=True)
    image[horse] = ReaderSensor(keyword='horse',label=True)
    image[ship] = ReaderSensor(keyword='ship',label=True)

    image['emb'] = ModuleLearner('pixels', module=ImageNetwork())
    image[animal] = ModuleLearner('emb', module=LinearNetwork())
    image[vehicle] = ModuleLearner('emb', module=LinearNetwork())
    image[airplane] = ModuleLearner('emb', module=LinearNetwork())
    image[dog] = ModuleLearner('emb', module=LinearNetwork())
    image[truck] = ModuleLearner('emb', module=LinearNetwork())
    image[automobile] = ModuleLearner('emb', module=LinearNetwork())
    image[bird] = ModuleLearner('emb', module=LinearNetwork())
    image[cat] = ModuleLearner('emb', module=LinearNetwork())
    image[deer] = ModuleLearner('emb', module=LinearNetwork())
    image[frog] = ModuleLearner('emb', module=LinearNetwork())
    image[horse] = ModuleLearner('emb', module=LinearNetwork())
    image[ship] = ModuleLearner('emb', module=LinearNetwork())

class CIFAR10_1(datasets.CIFAR10):

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR10_1, self).__init__(root, transform=transform, target_transform=target_transform, download=download)

        self.train = train  # training set or test set

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' + ' You can use download=True to download it')

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

                self.data.append(entry['data'][:100])
#               self.data.append(entry['data'])

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
        rDict = {}
        rDict['pixels'] = img
        animal_category = [2, 3, 4, 5, 6, 7]
        
        if target in animal_category:
            rDict['vehicle'] = [0]
            rDict['animal'] = [1]

        else:
            rDict['vehicle'] = [1]
            rDict['animal'] = [0]

        for i in range(10):
            rDict[target_dict[i]] = [0]
            
        rDict[target_dict[target]] = [1]
        
        return rDict

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

def setup_result_logger():
    import logging
    from logging.handlers import RotatingFileHandler
 
    logName = "CIFAR10"
    logLevel = logging.INFO
    import pathlib
    pathlib.Path("logs").mkdir(parents=True, exist_ok=True)
    logFilename='logs/cifar10Results.log'
    logFilesize=5*1024*1024
    logBackupCount=1
    logFileMode='a'

    logger = logging.getLogger(logName)

    # Create file handler and set level to info
    ch = RotatingFileHandler(logFilename, mode=logFileMode, maxBytes=logFilesize, backupCount=logBackupCount, encoding=None, delay=0)
    logger.setLevel(logLevel)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    # Add formatter to ch
    ch.setFormatter(formatter)

    # Add ch to logger
    logger.addHandler(ch)

    # Don't propagate
    logger.propagate = False
    print("Log file for %s is in: %s"%(logName,ch.baseFilename))
    
    return logger
        
def main():

    model_declaration()
    
    program = SolverPOIProgram(graph_ont, 
                               poi=(image, ), 
                               inferTypes=['ILP', 'local/argmax'], 
                               loss=MacroAverageTracker(NBCrossEntropyLoss()), 
                               metric={'ILP':PRF1Tracker(DatanodeCMMetric()),'softmax':PRF1Tracker(DatanodeCMMetric('local/argmax'))},
                               logger=setup_result_logger()
                               )

    ### load data
    val_size = 50
    # val_size = 5000

    trainset = load_cifar10(train=True)
    testset = load_cifar10(train=False)
    train_size = len(trainset) - val_size
    train_ds, val_ds = random_split(trainset, [train_size, val_size])
    print(len(train_ds), len(val_ds))

    program.train(training_set=train_ds, valid_set=val_ds, test_set=testset, train_epoch_num=2, Optim=lambda param: torch.optim.SGD(param, lr=.001))
    
    # program.save("/egr/research-hlr/elaheh/DomiKnowS/models/cifar_ILP")
    
    program.test(testset)
        
if __name__ == '__main__':
    main()