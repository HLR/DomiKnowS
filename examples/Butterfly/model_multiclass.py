import numpy as np
import torch.nn.functional as F
import torch
from torch import nn
from tqdm import tqdm
import torch.optim as optim
import regr
from regr.program import SolverPOIProgram, IMLProgram
from regr.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric
from regr.program.loss import BCEWithLogitsIMLoss, NBCrossEntropyLoss, NBCrossEntropyIMLoss
from graph_multiclass import graph
from regr.program.model.pytorch import SolverModel, IMLModel
from regr.program.primaldualprogram import PrimalDualProgram
from regr.sensor.pytorch.sensors import ReaderSensor
from regr.sensor.pytorch.learners import ModuleLearner
from torchvision.models.resnet import resnet50
from graph_multiclass import image_group_contains

output_size = 1000
from regr.sensor.pytorch.sensors import ReaderSensor, JointSensor, FunctionalSensor, FunctionalReaderSensor


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class ImageNetwork(torch.nn.Module):
    def __init__(self, n_outputs=2, model_size=10):
        super(ImageNetwork, self).__init__()
        self.num_outputs = n_outputs
        self.conv = resnet50(pretrained=True)
        set_parameter_requires_grad(self.conv, True)
        self.dense = nn.Linear(output_size, n_outputs)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, output_size)
        # x = self.dense(x)
        return x


class DenseNetwork(torch.nn.Module):
    def __init__(self, n_outputs=2):
        super(DenseNetwork, self).__init__()

        self.dense = nn.Linear(output_size, n_outputs)

    def forward(self, x):
        x = self.dense(x)
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
    family = graph['family']
    subFamily = graph['subFamily']

    image_group['pixels_group'] = ReaderSensor(keyword='pixels', device=device)
    image_group['family_group'] = ReaderSensor(keyword='family', device=device)
    image_group['subFamily_group'] = ReaderSensor(keyword='subFamily', device=device)

    def str_to_int_list(x):
        return torch.LongTensor([[int(i)] for i in x])

    def make_images(pixels_group, family_group, subFamily_group):
        return torch.ones((len(family_group.split("@@")), 1)), torch.squeeze(pixels_group, 0), str_to_int_list(
            family_group.split("@@")), str_to_int_list(subFamily_group.split("@@"))

    image[image_group_contains, "pixels", 'family_', "subFamily_"] = JointSensor(image_group['pixels_group'],
                                                                                 image_group["family_group"],
                                                                                 image_group["subFamily_group"],
                                                                                 forward=make_images)

    def label_reader(_, label):
        return label

    image[family] = FunctionalSensor(image_group_contains, "family_", forward=label_reader, label=True)
    image[subFamily] = FunctionalSensor(image_group_contains, "subFamily_", forward=label_reader, label=True)

    res = ImageNetwork(n_outputs=2, model_size=model_size)
    res.cuda(device)
    res.requires_grad_(True)
    image['emb'] = ModuleLearner('pixels', module=res, device=device)

    image[family] = ModuleLearner('emb', module=DenseNetwork(n_outputs=6), device=device)
    image[subFamily] = ModuleLearner('emb', module=DenseNetwork(n_outputs=21), device=device)

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
