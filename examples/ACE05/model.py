import torch
from transformers import BertTokenizer, BertModel

from regr.program import POIProgram
from regr.sensor.pytorch.sensors import ReaderSensor, ConstantSensor
from regr.sensor.pytorch.learners import ModuleLearner
from regr.sensor.pytorch.utils import UnBatchWrap

from sensors.tokenizers import Tokenizer


TRANSFORMER_MODEL = 'bert-base-uncased'


def model(graph, ):
    graph.detach()

    ling_graph = graph['linguistic']
    document = ling_graph['document']
    token = ling_graph['token']
    sentence_contains_word = document.relate_to(token)[0]

    document['index'] = ConstantSensor(data='John works for IBM.')  #ReaderSensor(keyword='text')
    tokenizer = BertTokenizer.from_pretrained(TRANSFORMER_MODEL)
    sentence_contains_word['forward'] = Tokenizer('index', mode='forward', to='index', tokenizer=tokenizer)

    # emb_model = BertModel.from_pretrained(TRANSFORMER_MODEL)
    class BERT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.module = BertModel.from_pretrained(TRANSFORMER_MODEL)
        def forward(self, input):
            input = input.unsqueeze(0)
            out, *_ = self.module(input)
            assert out.shape[0] == 1
            out = out.squeeze(0)
            return out
    # to freeze BERT, uncomment the following
    # for param in emb_model.base_model.parameters():
    #     param.requires_grad = False
    token['emb'] = ModuleLearner('index', module=BERT())
    program = POIProgram(graph, poi=(token['emb'],))

    return program
