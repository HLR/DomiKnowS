import sys

sys.path.append('.')

import torch

from regr.graph import Graph, Concept, Relation
from regr.graph import ifL, notL, andL, orL
from regr.program import LearningBasedProgram, POIProgram
from regr.sensor.pytorch.learners import ModuleLearner
from regr.sensor.pytorch.sensors import ReaderSensor
from regr.program.model.pytorch import PoiModel
from regr.program.metric import MacroAverageTracker, PRF1Tracker
from regr.program.loss import NBCrossEntropyLoss

from sensors.tweetSensor import SentenceRepSensor
from tweet_reader import SentimentReader


Graph.clear()
Concept.clear()
Relation.clear()

import logging
logging.basicConfig(level=logging.INFO)

def prediction_softmax(pr, gt):
  return torch.softmax(pr.data, dim=-1)

with Graph('example') as graph:

  twit= Concept(name = 'tweet')
  word = Concept (name = 'word')

  PositiveLabel = twit(name = 'PositiveLabel')
  NegativeLabel = twit(name ='NegativeLabel')

  (twit_contains_words,) = twit.contains(word)

  # ifL(PositiveLabel, notL(NegativeLabel))
  orL(andL(NegativeLabel, notL(PositiveLabel)), andL(PositiveLabel, notL(NegativeLabel)))

#Reading the data from a dictionary per learning example using reader sensors
twit['raw'] = ReaderSensor(keyword= 'tweet')
twit[PositiveLabel] = ReaderSensor(keyword='PositiveLabel', label = True)
twit[NegativeLabel] = ReaderSensor(keyword='NegativeLabel', label = True)
#Reading the features of each tweet using a sensor
twit['emb'] = SentenceRepSensor('raw')

#Associating the output lable with a learnig module
#If you have more features you need to concat them and introduce a new name before using them for the learner
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(96,128)
        self.l2 = torch.nn.Linear(128,2)
    def forward(self, x):
        a1 = self.l1(x)
        a1 = torch.nn.functional.relu(a1)
        a2 = self.l2(a1)
        return a2
twit[PositiveLabel] = ModuleLearner('emb', module = Net())
twit[NegativeLabel] = ModuleLearner('emb', module = Net())

#The reader will return the whole list of learning examples each of which is a dictionary
ReaderObjectsIterator = SentimentReader("examples/SentimentAnalysis/twitter_data/train5k.csv", "csv")

#The program takes the graph and learning approach as input
program = POIProgram(graph, loss=MacroAverageTracker(NBCrossEntropyLoss()), metric=PRF1Tracker())

#The program is ready to train:
for datanode in program.populate(dataset=list(ReaderObjectsIterator.run())[1:2]):
    print('datanode:', datanode)
    print('positive:', datanode.getAttribute(PositiveLabel).softmax(-1))
    print('negative:', datanode.getAttribute(NegativeLabel).softmax(-1))
    datanode.inferILPConstrains(fun=lambda val: torch.tensor(val).softmax(dim=-1).detach().cpu().numpy().tolist(), epsilon=None)
    print('inference positive:', datanode.getAttribute(PositiveLabel, 'ILP'))
    print('inference negative:', datanode.getAttribute(NegativeLabel, 'ILP'))

# program.populate(list(ReaderObjectsIterator.run())[1:2])
program.train(list(ReaderObjectsIterator.run()), train_epoch_num=30, Optim=torch.optim.Adam)
print(program.model.loss)
print(program.model.metric)

print('-'*40)

for _ in program.test(list(ReaderObjectsIterator.run())):
    pass
print(program.model.loss)
print(program.model.metric)
