from regr.sensor.sensor import Sensor
from Graphs.Sensors.mainSensors import ReaderSensor, CallingSensor
# from .mainSensors import ReaderSensor, CallingSensor
from typing import Dict, Any
from flair.embeddings import WordEmbeddings, CharacterEmbeddings, FlairEmbeddings, StackedEmbeddings, BertEmbeddings, ELMoEmbeddings
from flair.models import SequenceTagger
import torch
from sys import exit
import pdb
from flair.data import Sentence




class SentenceReaderSensor(CallingSensor):
    def forward(
        self,
        context: Dict[str, Any]
    ) -> Any:
        # return context[self.pres[0].fullname][0]
        return Sentence("I am highly motivated to capture the relationships of washington with berlin")


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


def translator(label):
    items = ['PRON', 'VERB', 'ADV', 'ADJ', 'PART', 'DET', 'NOUN', 'ADP']
    vector = torch.zeros(len(items))
    for item in range(len(items)):
        if label == items[item]:
            vector[item] = 1
    return vector.view(1, len(items))


class SentencePosTaggerSensor(SentenceSensor):
    def __init__(self, *pres, reader):
        super().__init__(*pres)
        self.reader = reader

    def forward(
            self,
            context: Dict[str, Any]
    ) -> Any:
        super(SentencePosTaggerSensor, self).forward(context=context)
        tagger = SequenceTagger.load('pos')
        tagger.predict(context[self.sentencesensor.fullname])
        _dict = context[self.sentencesensor.fullname].to_dict(tag_type='pos')
        _list = []
        for item in _dict['entities']:
            # _list.append(self.reader.posTagEncoder(item['type']))
            _list.append(translator(item['type']))
        return torch.stack(_list)
