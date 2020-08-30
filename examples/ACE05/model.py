import torch
from transformers import BertTokenizer, BertModel

from regr.program import POIProgram
from regr.sensor.pytorch.sensors import ReaderSensor, ConstantSensor, FunctionalSensor
from regr.sensor.pytorch.learners import ModuleLearner
from regr.sensor.pytorch.query_sensor import CandidateSensor, CandidateRelationSensor
from regr.sensor.pytorch.utils import UnBatchWrap

from sensors.tokenizers import Tokenizer


TRANSFORMER_MODEL = 'bert-base-uncased'


def cartesian_concat(*inputs):
    # torch cat is not broadcasting, do repeat manually
    input_iter = iter(inputs)
    output = next(input_iter)
    for input in input_iter:
        *dol, dof = output.shape
        *dil, dif = input.shape
        output = output.view(*dol, *(1,)*len(dil), dof).repeat(*(1,)*len(dol), *dil, 1)
        input = input.view(*(1,)*len(dol), *dil, dif).repeat(*dol, *(1,)*len(dil), 1)
        output = torch.cat((output, input), dim=-1)
    return output


def model(graph, ):
    graph.detach()

    ling_graph = graph['linguistic']
    document = ling_graph['document']
    token = ling_graph['token']
    span_candidate = ling_graph['span_candidate']
    span = ling_graph['span']
    document_contains_word = document.relate_to(token)[0]
    span_is_span_candidate = span.relate_to(span_candidate)[0]

    document['index'] = ConstantSensor(data='John works for IBM.')  #ReaderSensor(keyword='text')
    tokenizer = BertTokenizer.from_pretrained(TRANSFORMER_MODEL)
    document_contains_word['forward'] = Tokenizer('index', mode='forward', to='index', tokenizer=tokenizer)

    # emb_model = BertModel.from_pretrained(TRANSFORMER_MODEL)
    # to freeze BERT, uncomment the following
    # for param in emb_model.base_model.parameters():
    #     param.requires_grad = False
    class BERT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.module = BertModel.from_pretrained(TRANSFORMER_MODEL)
            # to freeze BERT, uncomment the following
            # for param in emb_model.base_model.parameters():
            #     param.requires_grad = False
        def forward(self, input):
            input = input.unsqueeze(0)
            out, *_ = self.module(input)
            assert out.shape[0] == 1
            out = out.squeeze(0)
            return out
    token['emb'] = ModuleLearner('index', module=BERT())
    def token_to_span_candidate(spans, start, end):
        length = end.instanceID - start.instanceID
        if length > 0 and length < 10:
            return True
        else:
            return False
    span_candidate['index'] = CandidateSensor(forward=token_to_span_candidate)
    def span_candidate_emb(token_emb, span_index):
        embs = cartesian_concat(token_emb, token_emb)
        span_index = span_index.rename(None)
        span_index = span_index.unsqueeze(-1).repeat(1, 1, embs.shape[-1])
        selected = embs.masked_select(span_index).view(-1, embs.shape[-1])
        return selected
    span_candidate['emb'] = FunctionalSensor(token['emb'], span_candidate['index'], forward=span_candidate_emb)
    span_candidate[span] = ModuleLearner('emb', module=torch.nn.Linear(768*2, 2))
    def span_candidate_to_span(spans, span_candidate, _):
        span_candidate
        return True
    span['index'] = CandidateRelationSensor(span_candidate[span], relations=(span_is_span_candidate,), forward=span_candidate_to_span)
    def span_candidate_emb(span_candidate_emb, span_index):
        selected = span_candidate_emb.masked_select(span_index).view(-1, span_candidate_emb.shape[-1])
        return selected
    span['emb'] = FunctionalSensor(span_candidate['emb'], span['index'], forward=span_candidate_emb)
    program = POIProgram(graph, poi=(span['emb'],))

    return program
