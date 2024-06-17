import numpy as np
import torch.nn.functional as F
import torch

from torch import nn
from tqdm import tqdm
import torch.optim as optim
import domiknows
from domiknows.program import SolverPOIProgram, IMLProgram
from domiknows.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric
from domiknows.program.loss import BCEWithLogitsIMLoss, NBCrossEntropyLoss, NBCrossEntropyIMLoss
from graph import graph
from domiknows.program.model.pytorch import SolverModel, IMLModel
from domiknows.program.lossprogram import PrimalDualProgram
from domiknows.sensor.pytorch.sensors import ReaderSensor
from domiknows.sensor.pytorch.learners import ModuleLearner
from torchvision.models.resnet import resnet50
from graph import image_group_contains

output_size = 45
from domiknows.sensor.pytorch.sensors import ReaderSensor, JointSensor, FunctionalSensor, FunctionalReaderSensor


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class ImageNetworkNormal(torch.nn.Module):
    def __init__(self, n_outputs=2, model_size=10):
        super(ImageNetwork, self).__init__()
        print("size of the model is: ", model_size)
        self.num_outputs = n_outputs
        self.conv1 = nn.Conv2d(3, model_size, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(model_size, model_size * 2, 5)
        self.conv3 = nn.Conv2d(model_size * 2, model_size, 5)
        self.conv4 = nn.Conv2d(model_size, 5, 3)
        self.bn = nn.BatchNorm2d(5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.bn(F.relu(self.conv4(x)))
        x = self.pool(x)
        x = x.view(-1, output_size)
        return x


output_size = 1000


class ImageNetwork(torch.nn.Module):
    def __init__(self, n_outputs=2, model_size=10):
        super(ImageNetwork, self).__init__()
        self.num_outputs = n_outputs
        self.conv = resnet50(pretrained=True)
        set_parameter_requires_grad(self.conv, False)
        self.dense = nn.Linear(output_size, n_outputs)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, output_size)
        return x


class DenseNetwork(torch.nn.Module):
    def __init__(self, n_outputs=2):
        super(DenseNetwork, self).__init__()

        self.dense1  = nn.Linear(output_size, 2)
        # self.dense2 = nn.Linear(output_size//2,output_size//4)
        # self.dense3 = nn.Linear(output_size//3,n_outputs)

    def forward(self, x):
        x = self.dense1(x)
        return x


def model_declaration(device, solver='iml', lambdaValue=0.5, model_size=10):
    solver = solver.lower()
    """
    this function creates and defines the structure of our graph model

    @return: a program based on the graph and it solver
    """
    graph.detach()
    image_group = graph['image_group']
    image = graph['image']
    animal = graph['animal']
    cat = graph['cat']
    dog = graph['dog']
    monkey = graph['monkey']
    squirrel = graph['squirrel']

    flower = graph['flower']
    daisy = graph['daisy']
    dandelion = graph['dandelion']
    rose = graph['rose']
    sunflower = graph['sunflower']
    tulip = graph['tulip']

    {'monkey': [], 'cat': [], 'squirrel': [], 'dog': [], 'daisy': [], 'dandelion': [], 'rose': [], 'tulip': [],
     'sunflower': [], 'flower': [], 'animal': [], 'pixels': []}
    image_group['pixels_group'] = ReaderSensor(keyword='pixels', device=device)
    image_group["animal_group"] = ReaderSensor(keyword='animal', device=device)
    image_group["cat_group"] = ReaderSensor(keyword='cat', device=device)
    image_group["dog_group"] = ReaderSensor(keyword='dog', device=device)
    image_group["monkey_group"] = ReaderSensor(keyword='monkey', device=device)
    image_group["squirrel_group"] = ReaderSensor(keyword='squirrel', device=device)
    image_group['flower_group'] = ReaderSensor(keyword='flower', device=device)
    image_group["daisy_group"] = ReaderSensor(keyword='daisy', device=device)
    image_group["dandelion_group"] = ReaderSensor(keyword='dandelion', device=device)
    image_group["rose_group"] = ReaderSensor(keyword='rose', device=device)
    image_group["sunflower_group"] = ReaderSensor(keyword='sunflower', device=device)
    image_group["tulip_group"] = ReaderSensor(keyword='tulip', device=device)

    def str_to_int_list(x):
        return torch.LongTensor([[int(i)] for i in x])

    def make_images(pixels_group, animal_group, cat_group, dog_group, monkey_group, squirrel_group, flower_group,
                    daisy_group, dandelion_group, rose_group, sunflower_group, tulip_group):
        # print("inside make images:",torch.squeeze(pixels_group,0) .shape,str_to_int_list(animal_group.split("@@")))
        return torch.ones((len(animal_group.split("@@")), 1)), torch.squeeze(pixels_group, 0), str_to_int_list(
            animal_group.split("@@")), str_to_int_list(cat_group.split("@@")), str_to_int_list(
            dog_group.split("@@")), str_to_int_list(monkey_group.split("@@")), str_to_int_list(
            squirrel_group.split("@@")), str_to_int_list(flower_group.split("@@")), str_to_int_list(
            daisy_group.split("@@")), str_to_int_list(dandelion_group.split("@@")), str_to_int_list(
            rose_group.split("@@")), str_to_int_list(sunflower_group.split("@@")), str_to_int_list(
            tulip_group.split("@@"))

    image[
        image_group_contains, "pixels", 'animal_', "cat_", "dog_", "monkey_", "squirrel_", "flower_", "daisy_", "dandelion_", "rose_", "sunflower_", "tulip_"] = JointSensor(
        image_group['pixels_group'], image_group["animal_group"], image_group["cat_group"], image_group["dog_group"],
        image_group["monkey_group"], image_group["squirrel_group"], image_group['flower_group'],
        image_group["daisy_group"], image_group["dandelion_group"], image_group["rose_group"],
        image_group["sunflower_group"], image_group["tulip_group"], forward=make_images)

    def label_reader(_, label):
        # print("label",label)
        return label

    image[animal] = FunctionalSensor(image_group_contains, "animal_", forward=label_reader, label=True)
    image[cat] = FunctionalSensor(image_group_contains, "cat_", forward=label_reader, label=True)
    image[dog] = FunctionalSensor(image_group_contains, "dog_", forward=label_reader, label=True)
    image[monkey] = FunctionalSensor(image_group_contains, "monkey_", forward=label_reader, label=True)
    image[squirrel] = FunctionalSensor(image_group_contains, "squirrel_", forward=label_reader, label=True)
    image[flower] = FunctionalSensor(image_group_contains, "flower_", forward=label_reader, label=True)
    image[daisy] = FunctionalSensor(image_group_contains, "daisy_", forward=label_reader, label=True)
    image[dandelion] = FunctionalSensor(image_group_contains, "dandelion_", forward=label_reader, label=True)
    image[rose] = FunctionalSensor(image_group_contains, "rose_", forward=label_reader, label=True)
    image[sunflower] = FunctionalSensor(image_group_contains, "sunflower_", forward=label_reader, label=True)
    image[tulip] = FunctionalSensor(image_group_contains, "tulip_", forward=label_reader, label=True)

    res=ImageNetwork(n_outputs=2, model_size=model_size)
    res.cuda(device)
    res.requires_grad_(False)
    image['emb'] = FunctionalSensor('pixels', forward=res,device=device)

    image[animal] = ModuleLearner('emb', module=DenseNetwork(), device=device)
    image[cat] = ModuleLearner('emb', module=DenseNetwork(), device=device)
    image[dog] = ModuleLearner('emb', module=DenseNetwork(), device=device)
    image[monkey] = ModuleLearner('emb', module=DenseNetwork(), device=device)
    image[squirrel] = ModuleLearner('emb', module=DenseNetwork(), device=device)
    image[flower] = ModuleLearner('emb', module=DenseNetwork(), device=device)
    image[daisy] = ModuleLearner('emb', module=DenseNetwork(), device=device)
    image[dandelion] = ModuleLearner('emb', module=DenseNetwork(), device=device)
    image[rose] = ModuleLearner('emb', module=DenseNetwork(), device=device)
    image[sunflower] = ModuleLearner('emb', module=DenseNetwork(), device=device)
    image[tulip] = ModuleLearner('emb', module=DenseNetwork(), device=device)

    if solver == 'iml':
        print("IMLProgram selected as solver")
        program = IMLProgram(graph, poi=(image,), inferTypes=['ILP', 'local/argmax'],
                             loss=MacroAverageTracker(BCEWithLogitsIMLoss(lmbd=lambdaValue)),
                             metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
                                     'softmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})
    elif solver == 'primal_dual':
        print("PrimalDualProgram + IML selected as solver")
        program = PrimalDualProgram(graph, SolverModel, poi=(image,), inferTypes=['ILP', 'local/argmax'],
                                    loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                    metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
                                            'softmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})
    elif solver == 'poi':
        print("POI Solver selected as solver")

        program = SolverPOIProgram(graph, loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                   metric=PRF1Tracker(DatanodeCMMetric('local/argmax')))

    return program
