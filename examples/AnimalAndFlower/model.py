import numpy as np
import torch.nn.functional as F
import torch
from torch import nn
from tqdm import tqdm
import torch.optim as optim

from regr.program import SolverPOIProgram, IMLProgram
from regr.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric
from regr.program.loss import BCEWithLogitsIMLoss
from graph import graph
from regr.sensor.pytorch.sensors import ReaderSensor
from regr.sensor.pytorch.learners import ModuleLearner
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
output_size = 45


class ImageNetwork(torch.nn.Module):
    def __init__(self, n_outputs=2):
        super(ImageNetwork, self).__init__()
        self.num_outputs = n_outputs
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 5, 5)
        self.conv4 = nn.Conv2d(5, 5, 3)
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


def model_declaration():
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

    image['emb'] = ModuleLearner('pixels', module=ImageNetwork(), device=device)
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
    program = IMLProgram(graph, poi=(image,), inferTypes=['ILP', 'local/argmax'],
                         loss=MacroAverageTracker(BCEWithLogitsIMLoss(lmbd=0.5)),
                         metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
                                 'softmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})
    return program



def train_with_single_network(train_set, validation_set, labelKeys):
    baseNet = ImageNetwork()
    classCounts = len(labelKeys)
    fcModels = {key: nn.Linear(output_size, 2) for key in labelKeys}

    criterion = nn.CrossEntropyLoss()
    params = list(baseNet.parameters())
    for key in labelKeys:
        params += list(fcModels.get(key).parameters())

    optimizer = optim.Adam(params)

    for epoch in range(1000):  # loop over the dataset multiple times

        for i, data in enumerate(tqdm(train_set), 0):
            optimizer.zero_grad()
            inputs = data['pixels']
            del data['pixels']
            losses = []
            baseNetResult = baseNet(inputs)
            for key in labelKeys:
                output = fcModels[key](baseNetResult)
                losses.append(criterion(output, torch.tensor([data[key]])))

            torch.autograd.backward(losses)
            optimizer.step()
        finalLabels = []
        finalRes = []

        for i, data in enumerate(tqdm(validation_set), 0):
            optimizer.zero_grad()
            inputs = data['pixels']
            del data['pixels']
            baseNetResult = baseNet(inputs)
            labels = []
            result = []
            for key in labelKeys:
                output = fcModels[key](baseNetResult)
                labels.append(data[key])
                result.append(int(torch.argmax(output)))

            finalRes.append(np.array(result))
            finalLabels.append(np.array(labels))

        finalForEach = [{"tp": 0, "tn": 0, "fp": 0, "fn": 0} for i in range(classCounts)]
        total = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        for results, labels in zip(finalRes, finalLabels):
            for index, (result, label) in enumerate(zip(results, labels)):
                if result == label and label == 1:
                    finalForEach[index]["tp"] += 1
                    total["tp"] += 1
                elif result == label and label == 0:
                    finalForEach[index]["tn"] += 1
                    total["tn"] += 1
                elif result != label and label == 1:
                    finalForEach[index]["fp"] += 1
                    total["fp"] += 1
                elif result != label and label == 0:
                    finalForEach[index]["fn"] += 1
                    total["fn"] += 1

        def getMetric(data):
            e = 0
            try:
                precision = (data["tp"]) / (data["tp"] + data["fp"])
            except:
                precision = 0
            try:
                recall = (data["tp"]) / (data["tp"] + data["fn"])
            except:
                recall = 0
            try:
                accuracy = (data["tp"] + data["tn"]) / (data["tp"] + data["tn"] + data["fp"] + data["fn"])
            except:
                accuracy = 0
            try:
                f1 = 2 * precision * recall / (precision + recall)
            except:
                f1 = 0
            return {"precision": precision, "recall": recall, "acc": accuracy, "f1": f1}

        print("")
        print("TOTAL", getMetric(total))
        for i in range(classCounts):
            label = list(data.keys())[i]
            print(label, finalForEach[i],getMetric(finalForEach[i]))
