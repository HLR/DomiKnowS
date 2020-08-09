import torch

from examples.SentimentAnalysis.sensors.tweetSensor import SentenceRepSensor
from examples.SentimentAnalysis.tweet_reader import SentimentReader
from regr.graph import Graph, Concept, Relation
from regr.program import LearningBasedProgram
from regr.sensor.torch.learner import ModuleLearner
from regr.sensor.pytorch.sensors import ReaderSensor
from regr.program.model.pytorch import PoiModel

Graph.clear()
Concept.clear()
Relation.clear()


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
twit['PositiveLabel'] = ReaderSensor(keyword='PositiveLabel', label = True)
twit['NegativeLabel'] = ReaderSensor(keyword='NegativeLabel', label = True)
#Reading the features of each tweet using a sensor
twit['emb'] = SentenceRepSensor('raw')

#Associating the output lable with a learnig module
twit['Label'] = ModuleLearner('emb', module = torch.nn.Linear(300,2))


#The reader will return the whole list of learning examples each of which is a dictionary
ReaderObjectsIterator = SentimentReader("twitter_data/train5k.csv", "csv")

#The program takes the graph and learning approach as input
program = LearningBasedProgram(graph, PoiModel)

#The program is ready to train:

program.train(list(ReaderObjectsIterator.run()))
print(program.model.loss)