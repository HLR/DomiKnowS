import torch

from examples.SentimentAnalysis.reader_to_jason import SentimentReader

from examples.SentimentAnalysis.sensors.tweetSensor import SentenceRepSensor
from regr.graph import Graph, Concept, Relation
from regr.program import LearningBasedProgram, POIProgram
from regr.program.loss import BCEWithLogitsLoss
from regr.program.metric import MacroAverageTracker, ValueTracker
from regr.sensor.torch.learner import ModuleLearner
from regr.sensor.pytorch.sensors import ReaderSensor

Graph.clear()
Concept.clear()
Relation.clear()


def prediction_softmax(pr, gt):
  return torch.softmax(pr.data, dim=-1)

with Graph('tweet') as graph:
  twit= Concept(name = 'twit')
  word = Concept (name = 'word')

  (twit_contains_words,) = twit.contains(word)


twit['raw'] = ReaderSensor(keyword= 'tweet')
twit['emb'] = SentenceRepSensor('raw')
twit['Label'] = ReaderSensor(keyword='PositiveLabel', label = True)
twit['Label'] = ModuleLearner('emb', module = torch.nn.Linear(300,2))


a = SentimentReader("twitter_data/train5k.csv", "csv")

program = POIProgram(graph,
            loss=MacroAverageTracker(BCEWithLogitsLoss()),
            metric=ValueTracker(prediction_softmax))
program.train(list(a.run()))
print(program.model.loss)