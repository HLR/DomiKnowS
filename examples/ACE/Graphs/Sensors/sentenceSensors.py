from regr.sensor.pytorch.sensors import TorchSensor, NominalSensor, ReaderSensor
# from .mainSensors import ReaderSensor, CallingSensor
from typing import Dict, Any
from flair.embeddings import WordEmbeddings, CharacterEmbeddings, FlairEmbeddings, StackedEmbeddings, BertEmbeddings, ELMoEmbeddings
from flair.models import SequenceTagger
import torch
from sys import exit
import pdb
from flair.data import Sentence


class SentenceSensor(TorchSensor):
    def __init__(self, *pres):
        super().__init__(*pres)
        self.sentence_value = pres[0]


class SentenceBertEmbedderSensor(SentenceSensor):
    def __init__(self, *pres):
        super().__init__(*pres)
        self.bert_embedding = BertEmbeddings()
 
    def forward(
        self,
    ) -> Any:
        self.bert_embedding.embed(self.fetch_value(self.sentence_value))
        return None


class SentenceGloveEmbedderSensor(SentenceSensor):
    def __init__(self, *pres):
        super().__init__(*pres)
        self.glove_embedding = WordEmbeddings('glove')
                                                        
    def forward(
            self,
    ) -> Any:
        self.glove_embedding.embed(self.fetch_value(self.sentence_value))
        return None


class SentenceFlairEmbedderSensor(SentenceSensor):
    def __init__(self, *pres):
        super().__init__(*pres)
        self.flair_embedding_backward = FlairEmbeddings('news-backward')
        
    def forward(
            self,
    ) -> Any:
        self.flair_embedding_backward.embed(self.fetch_value(self.sentence_value))
        return None


class FlairSentenceSensor(TorchSensor):
    def forward(self,) -> Any:
        return Sentence(self.inputs[0])

# def translator(label):
#     items = ['PRON', 'VERB', 'ADV', 'ADJ', 'PART', 'DET', 'NOUN', 'ADP']
#     vector = torch.zeros(len(items))
#     for item in range(len(items)):
#         if label == items[item]:
#             vector[item] = 1
#     return vector.view(1, len(items))
#
#


class SentencePosTagger(SentenceSensor):
    def __init__(self, *pres):
        super().__init__(*pres)
        self.tagger = SequenceTagger.load('pos')

    def forward(self,) -> Any:
        self.tagger.predict(self.inputs[0])
        return None


class TensorReaderSensor(ReaderSensor):
    def forward(
        self,
    ) -> Any:
        if self.data:
            try:
                    return torch.tensor(self.data[self.keyword], device=self.device)
            except:
                print("the key you requested from the reader doesn't exist")
                raise
        else:
            print("there is no data to operate on")
            raise Exception('not valid')
