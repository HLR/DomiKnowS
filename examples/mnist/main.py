import logging
import torch
from regr.program import LearningBasedProgram
from regr.sensor.pytorch.sensors import ReaderSensor
from regr.sensor.pytorch.learners import ModuleLearner
from model import MyModel, Net
import config
from time import time
from torchvision import datasets, transforms


def model_declaration(config):
    from graph import graph, image, digit

    graph.detach()

    image['pixels'] = ReaderSensor(keyword=0)
    image[digit] = ReaderSensor(keyword=1, label=True)
    image[digit] = ModuleLearner('pixels', module=Net(config.input_size, config.hidden_sizes, config.output_size))

    program = LearningBasedProgram(graph, MyModel)
    return program


def main():
    logging.basicConfig(level=logging.INFO)

    program = model_declaration(config)
    
    ### load data
    transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              torch.nn.Flatten(0)
                              ])
    trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
    testset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
    import random
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
        # sampler=random.sample(range(60000),600),
        shuffle=True,
        )
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
        # sampler=random.sample(range(10000),100),
        shuffle=True,
        )

    trainreader = trainloader
    testreader = testloader

    program.train(trainreader, test_set=testreader, train_epoch_num=config.epochs, Optim=lambda param: torch.optim.SGD(param, lr=0.01))
    # for loss, metric, x_node in program.test(data):
    #     print('loss:', loss)
    #     print(metric)
    #     # print('y0:', torch.softmax(x_node.getAttribute('<y0>'), dim=-1))
    #     # print('y0:', x_node.getAttribute('<y0>/ILP'))

if __name__ == '__main__':
    main()
