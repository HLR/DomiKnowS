from regr.sensor.sensor import Sensor
from Graphs.Sensors.mainSensors import ReaderSensor, CallingSensor
from typing import Dict, Any
from flair.embeddings import WordEmbeddings, CharacterEmbeddings, FlairEmbeddings, StackedEmbeddings, BertEmbeddings, ELMoEmbeddings
import torch
from sys import exit
import pdb
from flair.data import Sentence




class SentenceReaderSensor(ReaderSensor):
    def __init__(self,reader):
        super().__init__(reader)
    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        print("reader")
        return Sentence("I am here to be here")


class SentenceSensor(CallingSensor):
    def __init__(self, *pres):
        super().__init__(*pres)
        self.sentencesensor = pres[0]


class SentenceBertEmbedderSensor(SentenceSensor):
    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        super(SentenceBertEmbedderSensor, self).forward(context=context)
        bert_embedding = BertEmbeddings()
        bert_embedding.embed(context[self.sentencesensor.fullname])
        return None



# class SentencePosTagSensor(SentenceSensor):
#     def __init__(self, sentence):
#         super().__init__(sentence=sentence)


class SentenceGloveEmbedderSensor(SentenceSensor):
    def forward(
            self,
            context: Dict[str, Any]
    ) -> Any:
        super(SentenceGloveEmbedderSensor, self).forward(context=context)
        glove_embedding = WordEmbeddings('glove')
        glove_embedding.embed(context[self.sentencesensor.fullname])
        return None


class SentenceFlairEmbedderSensor(SentenceSensor):
    def forward(
            self,
            context: Dict[str, Any]
    ) -> Any:
        super(SentenceFlairEmbedderSensor, self).forward(context=context)
        flair_embedding_backward = FlairEmbeddings('news-backward')
        print("flair")
        flair_embedding_backward.embed(context[self.sentencesensor.fullname])
        return None


