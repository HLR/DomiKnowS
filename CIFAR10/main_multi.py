import sys
import torch
from domiknows.program import SolverPOIProgram

from domiknows.program.model.pytorch import PoiModel, IMLModel
from domiknows.program.model.lossModel import PrimalDualModel
from domiknows.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric
import matplotlib.pyplot as plt
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

class ImageModel(PrimalDualModel):
    def __init__(self, graph):
        super().__init__(
            graph,
            loss=MacroAverageTracker(NBCrossEntropyLoss()),
            metric=PRF1Tracker(DatanodeCMMetric()))
            #metric=PRF1Tracker())

from graph_multi import graph
graph.detach()
image = graph['image']
category = graph['category']
label = graph['tag']
    
def model_declaration():
    from domiknows.sensor.pytorch.sensors import ReaderSensor
    from domiknows.sensor.pytorch.learners import ModuleLearner
    from domiknows.program import LearningBasedProgram
    from torch import nn
    
    image['pixels'] = ReaderSensor(keyword='pixels')
    image[category] = ReaderSensor(keyword='category',label=True)
    image[label] = ReaderSensor(keyword='tag',label=True)

    image['emb'] = ModuleLearner('pixels', module=ImageNetwork())
    image[category] = ModuleLearner('emb', module=nn.Linear(16 * 5 * 5, 2))
    image[label] = ModuleLearner('emb', module=nn.Linear(16 * 5 * 5, 10))

    #program = LearningBasedProgram(graph, ImageModel)
    program = SolverPOIProgram(graph, loss=MacroAverageTracker(NBCrossEntropyLoss()), metric=PRF1Tracker(DatanodeCMMetric()))

    return program

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
        animal_category = [2,3,4,5,6,7]
        
        dict = {}
        dict['pixels'] = img
        category_dict = {0:'animal', 1: 'vehicle'}
        for i in range(10):
            dict['tag'] = [target] #[0] [1] [2] ... [9]

        if target in animal_category:
            dict['category'] = [0]
        else:
            dict['category'] = [1]

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
    trainset = load_cifar10(train=True)
    testset = load_cifar10(train=False)

    #print(trainset[0])

    program.train(trainset, train_epoch_num=10, Optim=lambda param: torch.optim.SGD(param, lr=.001))
    
    for datanode in program.populate(dataset=testset):
        #print('----------before ILP---------')
        
        label = graph['tag']
        
        #for l in label.values:
            #print(l, datanode.getAttribute(l).softmax(-1))
    
        datanode.inferILPResults(*category.values, *label.values, fun=None)
   
        print('\n----------after ILP---------')
        
        print("\n --- catagory")

        output = [category.get_value(i) + ": " + str(round(datanode.getAttribute(category)[i].item(), 2)) for i in range(len(category.values))]
        print("Output:   ", output)
                
        soft = [category.get_value(i) + ": " + str(round(datanode.getAttribute(category, "local", "softmax")[i].item(), 2)) for i in range(len(category.values))]
        print("Softmax:  ", soft)

        labelIndex = datanode.getAttribute(category, "label").item()
        print("Label:     %s"%(category.get_value(labelIndex)))
        
        for c in category.values:
            predt_category = datanode.getAttribute(c, 'ILP').item()
            if predt_category == 1.0:
                print("Inference: %s"%(c))
                
        print("\n --- tag")
          
        output = [label.get_value(i) + ": " + str(round(datanode.getAttribute(label)[i].item(), 2)) for i in range(len(label.values))]
        print("Output:   ", output)
                
        soft = [label.get_value(i) + ": " + str(round(datanode.getAttribute(label, "local", "softmax")[i].item(), 2)) for i in range(len(label.values))]
        print("Softmax:  ", soft)

        labelIndex = datanode.getAttribute(label, "label").item()
        print("Label:     %s"%(label.get_value(labelIndex)))

        prediction = ''
        for t in label.values:
            predt_label = datanode.getAttribute(t, 'ILP').item()
            if predt_label == 1.0:
                print("Inference: %s"%(t))
                prediction = t

        #d = datanode.getAttributes()['pixels'].numpy()
        #plt.figure()
        #plt.imshow((d[0,:,:]),interpolation='nearest', aspect='auto')
        #plt.text(5, 5, 'prediction: '+str(prediction), color='white',fontsize=15 )
        #plt.savefig(str(counter)+'.png')
        #plt.show()
        
        ILPmetrics = datanode.getInferMetrics()
        print("\nILP metrics Total %s"%(ILPmetrics['Total']))
        
        #break
if __name__ == '__main__':
    main()
