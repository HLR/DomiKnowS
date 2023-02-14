import sys
import torch
from data.reader import EmailSpamReader
from domiknows.program.model.pytorch import PoiModel, IMLModel
# from domiknows.program.metric import MacroAverageTracker, ValueTracker
from domiknows.program.metric import MacroAverageTracker, PRF1Tracker, ValueTracker

from torchvision import datasets, transforms
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
import os,pickle
import numpy as np
from domiknows.program.loss import NBCrossEntropyLoss

sys.path.append('.')
sys.path.append('../..')

def prediction_softmax(pr, gt):
    return torch.softmax(pr.data, dim=-1)

# image[‘emb] = ModuleLearner(‘pixels’)
# image[Felan] = ModuleLearner(‘emb’, net)
class ImageNetwork(torch.nn.Module):
    def __init__(self, n_outputs=2):
        super(ImageNetwork, self).__init__()
        self.num_outputs = n_outputs
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)


        self.fc = nn.Linear(16 * 5 * 5, n_outputs)

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
        x = self.fc(x)
        return F.softmax(x, dim=-1)

class ImageModel(PoiModel):
    def __init__(self, graph):
        super().__init__(
            graph,
            loss=MacroAverageTracker(NBCrossEntropyLoss()),
            metric=PRF1Tracker())

from graph import graph, airplane, dog, truck

def model_declaration():
    from domiknows.sensor.pytorch.sensors import ReaderSensor, ForwardEdgeSensor, ConstantSensor, ConcatSensor
    from domiknows.sensor.pytorch.learners import ModuleLearner
    # from domiknows.sensor.pytorch.relation_sensors import CandidateReaderSensor
    from domiknows.program import LearningBasedProgram
    from domiknows.program.model.pytorch import PoiModel
    import torch
    from torch import nn
    graph.detach()
    image = graph['image']
    airplane = graph['airplane']
    dog = graph['dog']
    truck = graph['truck']

    image['pixels'] = ReaderSensor(keyword='pixels')
    image[airplane] = ReaderSensor(keyword='airplane',label=True)
    image[dog] = ReaderSensor(keyword='dog',label=True)
    image[truck] = ReaderSensor(keyword='truck',label=True)

    image[airplane] = ModuleLearner('pixels', module=ImageNetwork())
    image[dog] = ModuleLearner('pixels', module=ImageNetwork())
    image[truck] = ModuleLearner('pixels', module=ImageNetwork())
    program = LearningBasedProgram(graph, ImageModel)

    return program

class CIFAR10_subset(datasets.CIFAR10):

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):

        super(CIFAR10_subset, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

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

                a = np.array(entry['labels'])
                selected_ix = []
                selected_ix.extend(np.where(a == 0)[0][:100])# airplane
                selected_ix.extend(np.where(a == 5)[0][:100])# dog
                selected_ix.extend(np.where(a == 9)[0][:100])# truck
                selected_data = [entry['data'][i] for i in range(len(entry['data'])) if i in selected_ix]
                selected_labels = [entry['labels'][i] for i in range(len(entry['labels'])) if i in selected_ix]
                self.data.append(selected_data)
                # self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(selected_labels)
                    # self.targets.extend(entry['labels'])
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

        dict = {}
        dict['pixels'] = img
        if target == 0:
            dict['airplane'] = [1]
            dict['dog'] = [0]
            dict['truck'] = [0]
        elif target == 5:
            dict['airplane'] = [0]
            dict['dog'] = [1]
            dict['truck'] = [0]
        elif target == 9:
            dict['airplane'] = [0]
            dict['dog'] = [0]
            dict['truck'] = [1]
        else:
            return None

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

    return CIFAR10_subset(root=root, train=train, transform=transform,download=False)

def main():
    program = model_declaration()

    ### load data
    trainset = load_cifar10(train=True)
    # testset = load_cifar10(train=False)

    program.train(trainset, train_epoch_num=2, Optim=lambda param: torch.optim.SGD(param, lr=.001))
    program.test(trainset)
    for datanode in program.populate(dataset=trainset):
        print('airplane:', datanode.getAttribute(airplane))
        print('dog:', datanode.getAttribute(dog))
        print('truck:', datanode.getAttribute(truck))


        # datanode.inferILPConstrains(fun=lambda val: torch.tensor(val).softmax(dim=-1).detach().cpu().numpy().tolist(),
        #                             epsilon=None)
        # print('inference airplane:', datanode.getAttribute(airplane, 'ILP'))
        # print('inference dog:', datanode.getAttribute(dog, 'ILP'))
        # print('inference truck:', datanode.getAttribute(truck, 'ILP'))


    # for loss, metric, x_node in program.test(trainset):
    #     print('loss:', loss)
    #     print(metric)
    #     # print('airplane:', torch.softmax(x_node.getAttribute('airplane'), dim=-1))
    #     # print('dog:', torch.softmax(x_node.getAttribute('dog'), dim=-1))
    #     # print('truck:', torch.softmax(x_node.getAttribute('truck'), dim=-1))
    #
    #     # print('y0:', x_node.getAttribute('<y0>/ILP'))


if __name__ == '__main__':
    main()
