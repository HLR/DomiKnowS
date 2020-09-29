import os
import sys
# Please change the root to an absolute or relative path to DomiKnowS root.
# In case relative path is used, consider the printed `CWD` as current working directory.
sys.path.append('.')
sys.path.append('../..')
from typing import Any
from tweet_reader import SentimentReader
from regr.sensor.pytorch.sensors import TorchSensor
from regr.sensor.pytorch.query_sensor import DataNodeSensor
import torch

from regr.graph import Graph, Concept, Relation
from regr.graph import ifL, notL, andL, orL
from regr.program import LearningBasedProgram, POIProgram
from regr.sensor.pytorch.learners import ModuleLearner
from regr.sensor.pytorch.sensors import ReaderSensor, ConstantSensor, FunctionalSensor
from regr.sensor.pytorch.relation_sensors import EdgeSensor
from regr.sensor.sensor import Sensor
from regr.program.model.pytorch import PoiModel
from regr.program.metric import MacroAverageTracker, PRF1Tracker
from regr.program.loss import NBCrossEntropyLoss
import spacy

Graph.clear()
with Graph('example') as graph:
    twit= Concept(name = 'tweet')
    word = Concept (name = 'word')
    clause = Concept(name= 'clause')

    PositiveLabel = twit(name = 'PositiveLabel')
    NegativeLabel = twit(name ='NegativeLabel')

    (twit_contains_words,) = twit.contains(word)
    (twit_contains_clause,) = twit.contains(clause)

    # ifL(PositiveLabel, notL(NegativeLabel))
    orL(andL(NegativeLabel, notL(PositiveLabel)), andL(PositiveLabel, notL(NegativeLabel)))
    
from regr.graph import DataNodeBuilder
data_item = DataNodeBuilder({"graph": graph})
twit['index'] = ConstantSensor(data="This is a dummy senetence but do not ignore dummy senetences!")
for sensor in twit['index'].find(Sensor):
    sensor(data_item)
    
class ButDetector(DataNodeSensor):
    def forward(self, instance, *inputs) -> Any:
        if " but " in instance.getAttribute('index').lower():
            return True
        else:
            return False

class ClauseGenerator(EdgeSensor):
    def forward(self, ) -> Any:
        return self.inputs[0].split("but ")


class SpacyVector(DataNodeSensor):
    def forward(self, instance, *inputs):
        return instance.getAttribute('index').vector
    
class SentenceRepSensor(DataNodeSensor):
    def __init__(self, *pres, edges=None, forward= None ,label=False, device='auto', spacy=None):
        super().__init__(*pres, edges=edges, forward=forward, label=label, device=device)
        if spacy:
            self.nlp = spacy
        else:
            raise ValueError("spacy should be instantiated")
    def forward(self, instance, *inputs) -> Any:
        text = self.nlp(instance.getAttribute(self.pres[0]))
        return torch.from_numpy(text.vector).to(device=self.device)
#         return text.vector

twit['but_presence'] = ButDetector()
for sensor in twit['but_presence'].find(Sensor):
    sensor(data_item)
clause['index'] = ClauseGenerator('index', relation=twit_contains_clause, mode="forward")
for sensor in clause['index'].find(Sensor):
    sensor(data_item)
    
large_spacy = spacy.load('en_core_web_lg')
twit['repr'] = SentenceRepSensor('index', spacy=large_spacy)
for sensor in twit['repr'].find(Sensor):
    sensor(data_item)
clause['repr'] = SentenceRepSensor('index', spacy=large_spacy)
for sensor in clause['repr'].find(Sensor):
    sensor(data_item)