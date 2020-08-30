import torch
import numpy as np
from transformers import BertTokenizer, BertModel

from regr.program import POIProgram
from regr.sensor.pytorch.sensors import ReaderSensor, ConstantSensor, FunctionalSensor
from regr.sensor.pytorch.query_sensor import CandidateSensor
from regr.sensor.pytorch.learners import ModuleLearner
from regr.sensor.pytorch.utils import UnBatchWrap
from test_regr.sensor.pytorch.sensors import TestSensor, TestEdgeSensor

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
    span = ling_graph['span']
    pair = ling_graph['pair']
    dct = document.relate_to(token)[0]
    sarg1, sarg2 = span.has_a()
    dcs = document.relate_to(span)[0]
    parg1, parg2 = pair.relate_to(span)
    
    entities_graph = graph['ACE05/Entities']
    PER = entities_graph['PER']
    ORG = entities_graph['ORG']

    events_graph = graph['ACE05/Events']
    be_born = events_graph['Be-Born']
    participant_argument = events_graph['Participant']

    # involve(be_born, participant_argument, Person=PER)
    # involve(be_born, attribute_argument, Time=timex2, Place=(GPE, LOC, FAC))

    def find_event_arg(event_type, role_arg_type=None, arg_type=None):
        from regr.graph.logicalConstrain import LogicalConstrain, ifL, orL, andL

        for _, constraint in events_graph.logicalConstrains.items():
            if not isinstance(constraint, ifL):
                continue
            role_arg, xy, implication = constraint.e
            if role_arg_type is not None and not role_arg.relate_to(role_arg_type):
                continue
            event_type_, x, *arg_implication = implication.e
            if event_type is not event_type_:
                continue
            if arg_type is None:
                yield role_arg
                continue
            if len(arg_implication) == 1:
                *concepts, y = arg_implication[0].e
            else:
                concept, y = arg_implication
                concepts = (concept,)
            if arg_type in concepts:
                yield role_arg
                continue
    be_born_participant_person = next(find_event_arg(be_born, participant_argument, PER))

    document['index'] = TestSensor(expected_outputs='John works for IBM .')
    dct['forward'] = TestEdgeSensor('index', to='index', mode='forward', expected_outputs=np.ones((1,5)))
    token['emb'] = TestSensor(expected_outputs=np.random.randn(5,300))
    # 10 spans in this example
    def token_to_span(spans, start, end, token_emb):
        length = end.instanceID - start.instanceID
        if length > 0 and length < 10:
            return True
        else:
            return False
    span['index'] = CandidateSensor(token['emb'], forward=token_to_span)
    def span_emb(token_emb, span_index):
        embs = cartesian_concat(token_emb, token_emb)
        span_index = span_index.rename(None)
        span_index = span_index.unsqueeze(-1).repeat(1, 1, embs.shape[-1])
        selected = embs.masked_select(span_index).view(-1, embs.shape[-1])
        return selected
    span['emb'] = FunctionalSensor(token['emb'], span['index'], forward=span_emb)
    # 100 pairs in this example
    pair['index'] = CandidateSensor(token['emb'], forward=lambda *_: True)
    def pair_emb(span_emb):
        embs = cartesian_concat(span_emb, span_emb)
        return embs.view(-1, embs.shape[-1])
    pair['emb'] = FunctionalSensor(span['emb'], forward=pair_emb)
    pair[be_born_participant_person] = ConstantSensor('emb', data=np.random.rand(100, 2))

    program = POIProgram(graph, poi=(
        # pair[be_born_participant_person],
        pair['emb'],
        pair['index'],
        span['emb'],
        span['index'],
        # document['index'],
        token['index'],
        ))

    return program
