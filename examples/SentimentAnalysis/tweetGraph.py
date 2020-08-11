import torch

from examples.SentimentAnalysis.sensors.tweetSensor import SentenceRepSensor
from examples.SentimentAnalysis.tweet_reader import SentimentReader
from regr.graph import Graph, Concept, Relation
from regr.program import LearningBasedProgram, POIProgram
from regr.sensor.pytorch.learners import ModuleLearner
from regr.sensor.pytorch.sensors import ReaderSensor
from regr.program.model.pytorch import PoiModel
from regr.program.metric import MacroAverageTracker, PRF1Tracker
from regr.program.loss import NBCrossEntropyLoss

Graph.clear()
Concept.clear()
Relation.clear()

import logging
logging.basicConfig(level=logging.INFO)

def prediction_softmax(pr, gt):
  return torch.softmax(pr.data, dim=-1)

with Graph('tweet') as graph:

  twit= Concept(name = 'tweet')
  word = Concept (name = 'word')

  PositiveLabel = twit(name = 'PositiveLabel')
  NegativeLabel = twit(name ='NegativeLabel')

  (twit_contains_words,) = twit.contains(word)

#Reading the data from a dictionary per learning example using reader sensors
twit['raw'] = ReaderSensor(keyword= 'tweet')
twit[PositiveLabel] = ReaderSensor(keyword='PositiveLabel', label = True)
twit[NegativeLabel] = ReaderSensor(keyword='NegativeLabel', label = True)
#Reading the features of each tweet using a sensor
twit['emb'] = SentenceRepSensor('raw')

#Associating the output lable with a learnig module
#If you have more features you need to concat them and introduce a new name before using them for the learner

twit[PositiveLabel] = ModuleLearner('emb', module = torch.nn.Linear(96,2))
twit[NegativeLabel] = ModuleLearner('emb', module = torch.nn.Linear(96,2))

#The reader will return the whole list of learning examples each of which is a dictionary
ReaderObjectsIterator = SentimentReader("twitter_data/train5k.csv", "csv")

#The program takes the graph and learning approach as input
program = POIProgram(graph, loss=MacroAverageTracker(NBCrossEntropyLoss()), metric=PRF1Tracker())


#The program is ready to train:
# for datanode in program.populate(dataset=list(ReaderObjectsIterator.run())[1:2]):
#     print(datanode)
program.train(ReaderObjectsIterator.run())

# program.populate(list(ReaderObjectsIterator.run())[1:2])
# program.train(list(ReaderObjectsIterator.run()))
# print(program.model.loss)