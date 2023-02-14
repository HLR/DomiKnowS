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
from domiknows.program.model.pytorch import SolverModel, IMLModel
from domiknows.program.primaldualprogram import PrimalDualProgram
from domiknows.sensor.pytorch.sensors import ReaderSensor
from domiknows.sensor.pytorch.learners import ModuleLearner
from torchvision.models.resnet import resnet50
from graph_multiclass import image_group_contains, graph

output_size = 45
from domiknows.sensor.pytorch.sensors import ReaderSensor, JointSensor, FunctionalSensor, FunctionalReaderSensor


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class ImageNetworkNormal(torch.nn.Module):
    def __init__(self, n_outputs=2, model_size=10):
        super(ImageNetwork, self).__init__()
        #print("size of the model is: ", model_size)
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


output_size = 2048 * 4 * 4


class ImageNetwork(torch.nn.Module):
    def __init__(self, n_outputs=2, model_size=10):
        super(ImageNetwork, self).__init__()
        self.num_outputs = n_outputs
        self.conv = resnet50(pretrained=True)
        set_parameter_requires_grad(self.conv, True)
        self.conv = nn.Sequential(*list(self.conv.children())[:-2])

        self.dense = nn.Linear(output_size, n_outputs)

    def forward(self, x):
        #print("ImageNetwork1", x.shape)
        x = self.conv(x)
        #print("ImageNetwork2",x.shape)
        x = x.view(x.shape[0], output_size)
        #print("ImageNetwork3",x.shape)
        return x


class DenseNetwork(torch.nn.Module):
    def __init__(self, n_outputs=2):
        super(DenseNetwork, self).__init__()

        self.dense1 = nn.Linear(output_size, n_outputs)
        # self.dense2 = nn.Linear(output_size//2,output_size//4)
        # self.dense3 = nn.Linear(output_size//3,n_outputs)

    def forward(self, x):
        #print("DenseNetwork1", x.shape)
        x=F.dropout(x,0.5)
        x = self.dense1(x)
        #print("DenseNetwork2", x.shape)
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
    category = graph['category']
    tag = graph['tag']

    image_group['pixels_group'] = ReaderSensor(keyword='pixels', device=device)
    image_group['category_group'] = ReaderSensor(keyword='category', device=device)
    image_group['tag_group'] = ReaderSensor(keyword='tag', device=device)

    def str_to_int_list(x):
        return torch.LongTensor([[int(i[1:-1])] for i in x])

    def make_images(pixels_group, category_group, tag_group):
        return torch.ones((len(category_group.split("@@")), 1)), torch.squeeze(pixels_group, 0), str_to_int_list(
            category_group.split("@@")), str_to_int_list(tag_group.split("@@"))

    image[image_group_contains, "pixels", 'category_', "tag_"] = JointSensor(image_group['pixels_group'],
                                                                             image_group["category_group"],
                                                                             image_group["tag_group"],
                                                                             forward=make_images)

    def label_reader(_, label):
        return label

    image[category] = FunctionalSensor(image_group_contains, "category_", forward=label_reader, label=True)
    image[tag] = FunctionalSensor(image_group_contains, "tag_", forward=label_reader, label=True)

    res = ImageNetwork(n_outputs=2, model_size=model_size)
    res.cuda(device)
    res.requires_grad_(True)
    image['emb'] = ModuleLearner('pixels', module=res, device=device)

    image[category] = ModuleLearner('emb', module=DenseNetwork(n_outputs= 2), device=device)
    image[tag] = ModuleLearner('emb', module=DenseNetwork(n_outputs= 9), device=device)

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
