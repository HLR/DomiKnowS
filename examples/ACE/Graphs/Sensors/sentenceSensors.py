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
        data_item: Dict[str, Any]
    ) -> Any:
        return data_item[self.pres[0].fullname][0]
        # return Sentence("I am highly motivated to capture the relationships of washington with berlin")


class SentenceSensor(CallingSensor):
    def __init__(self, *pres):
        super().__init__(*pres)
        self.sentencesensor = pres[0]


class SentenceBertEmbedderSensor(SentenceSensor):
    def __init__(self, *pres):
        super().__init__(*pres)
        self.bert_embedding = BertEmbeddings()
 
    def forward(
        self,
        data_item: Dict[str, Any]
    ) -> Any:
        super(SentenceBertEmbedderSensor, self).forward(data_item) 
        self.bert_embedding.embed(data_item[self.sentencesensor.fullname])
        return None



# class SentencePosTagSensor(SentenceSensor):
#     def __init__(self, sentence):
#         super().__init__(sentence=sentence)


class SentenceGloveEmbedderSensor(SentenceSensor):
    def __init__(self, *pres):
        super().__init__(*pres)
        self.glove_embedding = WordEmbeddings('glove')
                                                        
    def forward(
            self,
            data_item: Dict[str, Any]
    ) -> Any:
        super(SentenceGloveEmbedderSensor, self).forward(data_item)
        self.glove_embedding.embed(data_item[self.sentencesensor.fullname])
        return None


class SentenceFlairEmbedderSensor(SentenceSensor):
    def __init__(self, *pres):
        super().__init__(*pres)
        self.flair_embedding_backward = FlairEmbeddings('news-backward')
        
    def forward(
            self,
            data_item: Dict[str, Any]
    ) -> Any:
        super(SentenceFlairEmbedderSensor, self).forward(data_item)
        
        # print("flair")
        self.flair_embedding_backward.embed(data_item[self.sentencesensor.fullname])
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
        self.tagger = SequenceTagger.load('pos')
        
    def forward(
            self,
            data_item: Dict[str, Any]
    ) -> Any:
        super(SentencePosTaggerSensor, self).forward(data_item)
        self.tagger.predict(data_item[self.sentencesensor.fullname])
        _dict = data_item[self.sentencesensor.fullname].to_dict(tag_type='pos')
        _list = []

        for item in _dict['entities']:
            # _list.append(self.reader.posTagEncoder(item['type']))
            _list.append(self.reader.postagEncoder(item['type']))
        print(_list[-1].shape)
        return torch.stack(_list)
