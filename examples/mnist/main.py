import logging
import torch
# from graph import x
from regr.program import LearningBasedProgram
from regr.sensor.pytorch.sensors import ReaderSensor, TorchEdgeReaderSensor
from regr.sensor.pytorch.learners import ModuleLearner
from regr.graph import Property
from graph import graph
from model import MyModel, Net
from config import HYPER_PARAMETER
import torchvision
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

def model_declaration(hp):
    graph.detach()

    x = graph['x']
    y0 = graph['y0']

    x['x'] = ReaderSensor(keyword='x')
    x[y0] = ReaderSensor(keyword='y0', label=True)
    x[y0] = ModuleLearner('x', module=Net(hp['input_size'], hp['hidden_sizes'], hp['output_size']).cuda())

    program = LearningBasedProgram(graph, MyModel)
    return program


def main():
    logging.basicConfig(level=logging.INFO)

    program = model_declaration(HYPER_PARAMETER)
    
    ### load data
    transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
    trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
    valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=512, shuffle=True)
    # trainloader = torch.utils.data.DataLoader(trainset, shuffle=True)
    # valloader = torch.utils.data.DataLoader(valset, shuffle=True)
    
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    print(images.shape)
    print(labels.shape)
    # import sys
    # sys.exit()

    data = [{
        'x': images.view(-1, 784),
        'y0': labels,
        }]

    print(data)

    program.train(data, train_epoch_num=HYPER_PARAMETER['epochs'], Optim=lambda param: torch.optim.SGD(param, lr=1))
    for loss, metric, x_node in program.test(data):
        print('loss:', loss)
        print(metric)
        # print('y0:', torch.softmax(x_node.getAttribute('<y0>'), dim=-1))
        # print('y0:', x_node.getAttribute('<y0>/ILP'))

if __name__ == '__main__':
    main()
