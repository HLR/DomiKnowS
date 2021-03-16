import wget
import os
import torch

from regr.program import POIProgram

from torch import nn
from regr.program.metric import MacroAverageTracker, PRF1Tracker
from regr.program.loss import NBCrossEntropyLoss

from regr.sensor.pytorch.learners import ModuleLearner
from regr.sensor.pytorch.sensors import ReaderSensor, TorchEdgeSensor, JointSensor, FunctionalSensor, \
    FunctionalReaderSensor

print('Downloading dataset')

url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
url_2 = "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt"

if not os.path.exists('data/squad1.1/train-v1.1.json'):
    os.mkdir("data/squad1.1")
    wget.download(url, 'data/squad1.1/train-v1.1.json')

if not os.path.exists('data/squad1.1/bert-base-uncased-vocab.txt'):
    wget.download(url_2, 'data/squad1.1/bert-base-uncased-vocab.txt')

from read_QA import QA_reader, DariusQABERT

out = QA_reader(max_lenght=256, sample_num=2)  # out = X_tokens, Y, Y_S, Y_E, Masks, SentNumber

from regr.graph import Graph, Concept, Relation
from regr.graph import ifL, notL, andL, orL, nandL

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('QA_graph') as graph:
    sentence = Concept(name='sentence')

reader = []
for x, y, y_s, y_e, mask, sent_numb in zip(*out):
    reader.append({"x": x, "y": y, "y_s": y_s, "y_e": y_e, "mask": mask, "sent_numb": sent_numb})

sentence["x"] = ReaderSensor(keyword='x')
sentence["y"] = ReaderSensor(keyword='y')
sentence["y_s"] = ReaderSensor(keyword='y_s')
sentence["y_e"] = ReaderSensor(keyword='y_e')
sentence["mask"] = ReaderSensor(keyword='mask')
sentence["sent_numb"] = ReaderSensor(keyword='sent_numb')


def put_near(x1, x2):
    return (x1, x2)


sentence['ans'] = FunctionalSensor("y_s","y_e", forward=put_near, label=True)

sentence["ans"] = ModuleLearner("x", "mask", "sent_numb", module=DariusQABERT())


class QACrossEntropy(nn.CrossEntropyLoss):
    def forward(self, input, target, *args, **kwargs):
        #print("in loss")
        #print(input.shape)
        #print(target)
        x=input
        (ys,ye)=target
        #print(ys)
        #print(ye)

        return super().forward(x[:,:,0], ys.unsqueeze(dim=0), *args, **kwargs)+super().forward(x[:,:,1], ye.unsqueeze(dim=0), *args, **kwargs)


program = POIProgram(graph, loss=MacroAverageTracker(QACrossEntropy()))

import logging

device = 'auto'
logging.basicConfig(level=logging.INFO)

program.train(reader, train_epoch_num=1, Optim=torch.optim.Adam, device=device)
print('Training result:')
print(program.model.loss)
print(program.model.metric)
