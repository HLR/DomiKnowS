import torch
from transformers import BertTokenizer

from regr.program import POIProgram
from regr.sensor.pytorch.sensors import ReaderSensor, ConstantSensor

from sensors.tokenizers import Tokenizer

def model(graph, ):
    graph.detach()

    ling_graph = graph['linguistic']
    sentence = ling_graph['sentence']
    word = ling_graph['word']
    sentence_contains_word = sentence.relate_to(word)[0]

    sentence['index'] = ConstantSensor(data='John works for IBM.')  #ReaderSensor(keyword='text')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    sentence_contains_word['forward'] = Tokenizer('index', mode='forward', to='index', tokenizer=tokenizer)
    program = POIProgram(graph, poi=(word['index'],))

    return program
