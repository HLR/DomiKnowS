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
from torchvision.models.resnet import resnet50
from graph import image_group_contains

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
        x = self.dense(x)
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

    coliadinae = graph['coliadinae']
    image_group["coliadinae_group"] = ReaderSensor(keyword='coliadinae', device=device)

    dismorphiinae = graph['dismorphiinae']
    image_group["dismorphiinae_group"] = ReaderSensor(keyword='dismorphiinae', device=device)

    pierinae = graph['pierinae']
    image_group["pierinae_group"] = ReaderSensor(keyword='pierinae', device=device)

    polyommatinae = graph['polyommatinae']
    image_group["polyommatinae_group"] = ReaderSensor(keyword='polyommatinae', device=device)

    theclinae = graph['theclinae']
    image_group["theclinae_group"] = ReaderSensor(keyword='theclinae', device=device)

    lycaeninae = graph['lycaeninae']
    image_group["lycaeninae_group"] = ReaderSensor(keyword='lycaeninae', device=device)

    aphnaeinae = graph['aphnaeinae']
    image_group["aphnaeinae_group"] = ReaderSensor(keyword='aphnaeinae', device=device)

    charaxinae = graph['charaxinae']
    image_group["charaxinae_group"] = ReaderSensor(keyword='charaxinae', device=device)

    limenitidinae = graph['limenitidinae']
    image_group["limenitidinae_group"] = ReaderSensor(keyword='limenitidinae', device=device)

    libytheinae = graph['libytheinae']
    image_group["libytheinae_group"] = ReaderSensor(keyword='libytheinae', device=device)

    danainae = graph['danainae']
    image_group["danainae_group"] = ReaderSensor(keyword='danainae', device=device)

    nymphalinae = graph['nymphalinae']
    image_group["nymphalinae_group"] = ReaderSensor(keyword='nymphalinae', device=device)

    apaturinae = graph['apaturinae']
    image_group["apaturinae_group"] = ReaderSensor(keyword='apaturinae', device=device)

    satyrinae = graph['satyrinae']
    image_group["satyrinae_group"] = ReaderSensor(keyword='satyrinae', device=device)

    heliconiinae = graph['heliconiinae']
    image_group["heliconiinae_group"] = ReaderSensor(keyword='heliconiinae', device=device)

    pyrginae = graph['pyrginae']
    image_group["pyrginae_group"] = ReaderSensor(keyword='pyrginae', device=device)

    hesperiinae = graph['hesperiinae']
    image_group["hesperiinae_group"] = ReaderSensor(keyword='hesperiinae', device=device)

    heteropterinae = graph['heteropterinae']
    image_group["heteropterinae_group"] = ReaderSensor(keyword='heteropterinae', device=device)

    parnassiinae = graph['parnassiinae']
    image_group["parnassiinae_group"] = ReaderSensor(keyword='parnassiinae', device=device)

    papilioninae = graph['papilioninae']
    image_group["papilioninae_group"] = ReaderSensor(keyword='papilioninae', device=device)

    nemeobiinae = graph['nemeobiinae']
    image_group["nemeobiinae_group"] = ReaderSensor(keyword='nemeobiinae', device=device)

    pieridae = graph['pieridae']
    image_group["pieridae_group"] = ReaderSensor(keyword='pieridae', device=device)

    lycaenidae = graph['lycaenidae']
    image_group["lycaenidae_group"] = ReaderSensor(keyword='lycaenidae', device=device)

    nymphalidae = graph['nymphalidae']
    image_group["nymphalidae_group"] = ReaderSensor(keyword='nymphalidae', device=device)

    hesperiidae = graph['hesperiidae']
    image_group["hesperiidae_group"] = ReaderSensor(keyword='hesperiidae', device=device)

    papilionidae = graph['papilionidae']
    image_group["papilionidae_group"] = ReaderSensor(keyword='papilionidae', device=device)

    riodinidae = graph['riodinidae']
    image_group["riodinidae_group"] = ReaderSensor(keyword='riodinidae', device=device)

    image_group['pixels_group'] = ReaderSensor(keyword='pixels', device=device)

    def str_to_int_list(x):
        return torch.LongTensor([[int(i)] for i in x])

    def make_images(pixels_group, coliadinae_group, dismorphiinae_group, pierinae_group, polyommatinae_group,
                    theclinae_group, lycaeninae_group, aphnaeinae_group, charaxinae_group, limenitidinae_group,
                    libytheinae_group, danainae_group, nymphalinae_group, apaturinae_group, satyrinae_group,
                    heliconiinae_group, pyrginae_group, hesperiinae_group, heteropterinae_group, parnassiinae_group,
                    papilioninae_group, nemeobiinae_group, pieridae_group, lycaenidae_group, nymphalidae_group,
                    hesperiidae_group, papilionidae_group, riodinidae_group):
        # print("inside make images:",torch.squeeze(pixels_group,0) .shape,str_to_int_list(animal_group.split("@@")))
        return torch.ones((len(coliadinae_group.split("@@")), 1)), torch.squeeze(pixels_group, 0), str_to_int_list(
            coliadinae_group.split("@@")), str_to_int_list(dismorphiinae_group.split("@@")), str_to_int_list(
            pierinae_group.split("@@")), str_to_int_list(polyommatinae_group.split("@@")), str_to_int_list(
            theclinae_group.split("@@")), str_to_int_list(lycaeninae_group.split("@@")), str_to_int_list(
            aphnaeinae_group.split("@@")), str_to_int_list(charaxinae_group.split("@@")), str_to_int_list(
            limenitidinae_group.split("@@")), str_to_int_list(libytheinae_group.split("@@")), str_to_int_list(
            danainae_group.split("@@")), str_to_int_list(nymphalinae_group.split("@@")), str_to_int_list(
            apaturinae_group.split("@@")), str_to_int_list(satyrinae_group.split("@@")), str_to_int_list(
            heliconiinae_group.split("@@")), str_to_int_list(pyrginae_group.split("@@")), str_to_int_list(
            hesperiinae_group.split("@@")), str_to_int_list(heteropterinae_group.split("@@")), str_to_int_list(
            parnassiinae_group.split("@@")), str_to_int_list(papilioninae_group.split("@@")), str_to_int_list(
            nemeobiinae_group.split("@@")), str_to_int_list(pieridae_group.split("@@")), str_to_int_list(
            lycaenidae_group.split("@@")), str_to_int_list(nymphalidae_group.split("@@")), str_to_int_list(
            hesperiidae_group.split("@@")), str_to_int_list(papilionidae_group.split("@@")), str_to_int_list(
            riodinidae_group.split("@@"))

    image[
        image_group_contains, "pixels", 'coliadinae_', 'dismorphiinae_', 'pierinae_', 'polyommatinae_', 'theclinae_',
        'lycaeninae_', 'aphnaeinae_', 'charaxinae_', 'limenitidinae_', 'libytheinae_', 'danainae_', 'nymphalinae_',
        'apaturinae_', 'satyrinae_', 'heliconiinae_', 'pyrginae_', 'hesperiinae_', 'heteropterinae_', 'parnassiinae_',
        'papilioninae_', 'nemeobiinae_', 'pieridae_', 'lycaenidae_', 'nymphalidae_', 'hesperiidae_', 'papilionidae_', 'riodinidae_'] = JointSensor(
        image_group['pixels_group'], image_group["coliadinae_group"], image_group["dismorphiinae_group"],
        image_group["pierinae_group"],
        image_group["polyommatinae_group"], image_group["theclinae_group"], image_group["lycaeninae_group"],
        image_group["aphnaeinae_group"], image_group["charaxinae_group"], image_group["limenitidinae_group"],
        image_group["libytheinae_group"], image_group["danainae_group"], image_group["nymphalinae_group"],
        image_group["apaturinae_group"], image_group["satyrinae_group"], image_group["heliconiinae_group"],
        image_group["pyrginae_group"], image_group["hesperiinae_group"], image_group["heteropterinae_group"],
        image_group["parnassiinae_group"], image_group["papilioninae_group"], image_group["nemeobiinae_group"],
        image_group["pieridae_group"], image_group["lycaenidae_group"], image_group["nymphalidae_group"],
        image_group["hesperiidae_group"], image_group["papilionidae_group"], image_group["riodinidae_group"],
        forward=make_images)

    def label_reader(_, label):
        # print("label",label)
        return label

    image[coliadinae] = FunctionalSensor(image_group_contains, "coliadinae_", forward=label_reader, label=True)
    image[dismorphiinae] = FunctionalSensor(image_group_contains, "dismorphiinae_", forward=label_reader, label=True)
    image[pierinae] = FunctionalSensor(image_group_contains, "pierinae_", forward=label_reader, label=True)
    image[polyommatinae] = FunctionalSensor(image_group_contains, "polyommatinae_", forward=label_reader, label=True)
    image[theclinae] = FunctionalSensor(image_group_contains, "theclinae_", forward=label_reader, label=True)
    image[lycaeninae] = FunctionalSensor(image_group_contains, "lycaeninae_", forward=label_reader, label=True)
    image[aphnaeinae] = FunctionalSensor(image_group_contains, "aphnaeinae_", forward=label_reader, label=True)
    image[charaxinae] = FunctionalSensor(image_group_contains, "charaxinae_", forward=label_reader, label=True)
    image[limenitidinae] = FunctionalSensor(image_group_contains, "limenitidinae_", forward=label_reader, label=True)
    image[libytheinae] = FunctionalSensor(image_group_contains, "libytheinae_", forward=label_reader, label=True)
    image[danainae] = FunctionalSensor(image_group_contains, "danainae_", forward=label_reader, label=True)
    image[nymphalinae] = FunctionalSensor(image_group_contains, "nymphalinae_", forward=label_reader, label=True)
    image[apaturinae] = FunctionalSensor(image_group_contains, "apaturinae_", forward=label_reader, label=True)
    image[satyrinae] = FunctionalSensor(image_group_contains, "satyrinae_", forward=label_reader, label=True)
    image[heliconiinae] = FunctionalSensor(image_group_contains, "heliconiinae_", forward=label_reader, label=True)
    image[pyrginae] = FunctionalSensor(image_group_contains, "pyrginae_", forward=label_reader, label=True)
    image[hesperiinae] = FunctionalSensor(image_group_contains, "hesperiinae_", forward=label_reader, label=True)
    image[heteropterinae] = FunctionalSensor(image_group_contains, "heteropterinae_", forward=label_reader, label=True)
    image[parnassiinae] = FunctionalSensor(image_group_contains, "parnassiinae_", forward=label_reader, label=True)
    image[papilioninae] = FunctionalSensor(image_group_contains, "papilioninae_", forward=label_reader, label=True)
    image[nemeobiinae] = FunctionalSensor(image_group_contains, "nemeobiinae_", forward=label_reader, label=True)
    image[pieridae] = FunctionalSensor(image_group_contains, "pieridae_", forward=label_reader, label=True)
    image[lycaenidae] = FunctionalSensor(image_group_contains, "lycaenidae_", forward=label_reader, label=True)
    image[nymphalidae] = FunctionalSensor(image_group_contains, "nymphalidae_", forward=label_reader, label=True)
    image[hesperiidae] = FunctionalSensor(image_group_contains, "hesperiidae_", forward=label_reader, label=True)
    image[papilionidae] = FunctionalSensor(image_group_contains, "papilionidae_", forward=label_reader, label=True)
    image[riodinidae] = FunctionalSensor(image_group_contains, "riodinidae_", forward=label_reader, label=True)

    image[coliadinae] = ModuleLearner('pixels', module=ImageNetwork(), device=device)
    image[dismorphiinae] = ModuleLearner('pixels', module=ImageNetwork(), device=device)
    image[pierinae] = ModuleLearner('pixels', module=ImageNetwork(), device=device)
    image[polyommatinae] = ModuleLearner('pixels', module=ImageNetwork(), device=device)
    image[theclinae] = ModuleLearner('pixels', module=ImageNetwork(), device=device)
    image[lycaeninae] = ModuleLearner('pixels', module=ImageNetwork(), device=device)
    image[aphnaeinae] = ModuleLearner('pixels', module=ImageNetwork(), device=device)
    image[charaxinae] = ModuleLearner('pixels', module=ImageNetwork(), device=device)
    image[limenitidinae] = ModuleLearner('pixels', module=ImageNetwork(), device=device)
    image[libytheinae] = ModuleLearner('pixels', module=ImageNetwork(), device=device)
    image[danainae] = ModuleLearner('pixels', module=ImageNetwork(), device=device)
    image[nymphalinae] = ModuleLearner('pixels', module=ImageNetwork(), device=device)
    image[apaturinae] = ModuleLearner('pixels', module=ImageNetwork(), device=device)
    image[satyrinae] = ModuleLearner('pixels', module=ImageNetwork(), device=device)
    image[heliconiinae] = ModuleLearner('pixels', module=ImageNetwork(), device=device)
    image[pyrginae] = ModuleLearner('pixels', module=ImageNetwork(), device=device)
    image[hesperiinae] = ModuleLearner('pixels', module=ImageNetwork(), device=device)
    image[heteropterinae] = ModuleLearner('pixels', module=ImageNetwork(), device=device)
    image[parnassiinae] = ModuleLearner('pixels', module=ImageNetwork(), device=device)
    image[papilioninae] = ModuleLearner('pixels', module=ImageNetwork(), device=device)
    image[nemeobiinae] = ModuleLearner('pixels', module=ImageNetwork(), device=device)
    image[pieridae] = ModuleLearner('pixels', module=ImageNetwork(), device=device)
    image[lycaenidae] = ModuleLearner('pixels', module=ImageNetwork(), device=device)
    image[nymphalidae] = ModuleLearner('pixels', module=ImageNetwork(), device=device)
    image[hesperiidae] = ModuleLearner('pixels', module=ImageNetwork(), device=device)
    image[papilionidae] = ModuleLearner('pixels', module=ImageNetwork(), device=device)
    image[riodinidae] = ModuleLearner('pixels', module=ImageNetwork(), device=device)

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
