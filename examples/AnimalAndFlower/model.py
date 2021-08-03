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
from graph import graph
from regr.program.model.pytorch import SolverModel, IMLModel
from regr.program.primaldualprogram import PrimalDualProgram
from regr.sensor.pytorch.sensors import ReaderSensor
from regr.sensor.pytorch.learners import ModuleLearner

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
output_size = 45


class ImageNetwork(torch.nn.Module):
    def __init__(self, n_outputs=2,model_size=10):
        super(ImageNetwork, self).__init__()
        self.num_outputs = n_outputs
        self.conv1 = nn.Conv2d(3, model_size, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(model_size, model_size*2, 5)
        self.conv3 = nn.Conv2d(model_size*2, model_size, 5)
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


def model_declaration(solver='iml',lambdaValue=0.5,model_size=10):
    solver = solver.lower()
    """
    this function creates and defines the structure of our graph model

    @return: a program based on the graph and it solver
    """
    graph.detach()
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

    image['pixels'] = ReaderSensor(keyword='pixels', device=device)
    image[animal] = ReaderSensor(keyword='animal', label=True, device=device)
    image[cat] = ReaderSensor(keyword='cat', label=True, device=device)
    image[dog] = ReaderSensor(keyword='dog', label=True, device=device)
    image[monkey] = ReaderSensor(keyword='monkey', label=True, device=device)
    image[squirrel] = ReaderSensor(keyword='squirrel', label=True, device=device)
    image[flower] = ReaderSensor(keyword='flower', label=True, device=device)
    image[daisy] = ReaderSensor(keyword='daisy', label=True, device=device)
    image[dandelion] = ReaderSensor(keyword='dandelion', label=True, device=device)
    image[rose] = ReaderSensor(keyword='rose', label=True, device=device)
    image[sunflower] = ReaderSensor(keyword='sunflower', label=True, device=device)
    image[tulip] = ReaderSensor(keyword='tulip', label=True, device=device)

    image['emb'] = ModuleLearner('pixels', module=ImageNetwork(model_size), device=device)
    image[animal] = ModuleLearner('emb', module=nn.Linear(output_size, 2), device=device)
    image[cat] = ModuleLearner('emb', module=nn.Linear(output_size, 2), device=device)
    image[dog] = ModuleLearner('emb', module=nn.Linear(output_size, 2), device=device)
    image[monkey] = ModuleLearner('emb', module=nn.Linear(output_size, 2), device=device)
    image[squirrel] = ModuleLearner('emb', module=nn.Linear(output_size, 2), device=device)
    image[flower] = ModuleLearner('emb', module=nn.Linear(output_size, 2), device=device)
    image[daisy] = ModuleLearner('emb', module=nn.Linear(output_size, 2), device=device)
    image[dandelion] = ModuleLearner('emb', module=nn.Linear(output_size, 2), device=device)
    image[rose] = ModuleLearner('emb', module=nn.Linear(output_size, 2), device=device)
    image[sunflower] = ModuleLearner('emb', module=nn.Linear(output_size, 2), device=device)
    image[tulip] = ModuleLearner('emb', module=nn.Linear(output_size, 2), device=device)

    if solver == 'iml':
        print("IMLProgram selected as solver")
        program = IMLProgram(graph, poi=(image,), inferTypes=['ILP', 'local/argmax'],
                             loss=MacroAverageTracker(BCEWithLogitsIMLoss(lmbd=lambdaValue)),
                             metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
                                     'softmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})
    elif solver == 'primal_dual':
        print("PrimalDualProgram + IML selected as solver")
        program = PrimalDualProgram(graph, IMLModel, poi=(image,), inferTypes=['ILP', 'local/argmax'],
                                    loss=MacroAverageTracker(BCEWithLogitsIMLoss(lmbd=lambdaValue)),
                                    metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
                                            'softmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})
    elif solver == 'poi':
        print("POI Solver selected as solver")

        program = SolverPOIProgram(graph, loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                   metric=PRF1Tracker(DatanodeCMMetric('local/argmax')))

    return program

