from regr.sensor.pytorch.sensors import TorchSensor
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
# class SentencePosTaggerSensor(SentenceSensor):
#     def __init__(self, *pres, reader):
#         super().__init__(*pres)
#         self.reader = reader
#         self.tagger = SequenceTagger.load('pos')
#
#     def forward(
#             self,
#             context: Dict[str, Any]
#     ) -> Any:
#         super(SentencePosTaggerSensor, self).forward(context=context)
#         self.tagger.predict(context[self.sentencesensor.fullname])
#         _dict = context[self.sentencesensor.fullname].to_dict(tag_type='pos')
#         _list = []
#
#         for item in _dict['entities']:
#             # _list.append(self.reader.posTagEncoder(item['type']))
#             _list.append(self.reader.postagEncoder(item['type']))
#         print(_list[-1].shape)
#         return torch.stack(_list)
