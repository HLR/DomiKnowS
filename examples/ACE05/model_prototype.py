import torch
import numpy as np

from regr.program import POIProgram
from regr.sensor.pytorch.sensors import FunctionalSensor
from regr.sensor.pytorch.query_sensor import CandidateSensor
from test_regr.sensor.pytorch.sensors import TestSensor, TestLearner, TestEdgeSensor

from models import cartesian_concat, find_event_arg


def model(graph, ):
    graph.detach()

    ling_graph = graph['linguistic']
    document = ling_graph['document']
    span = ling_graph['span']
    pair = ling_graph['pair']
    dcs = document.relate_to(span)[0]
    parg1, parg2 = pair.relate_to(span)
    

    # example event constraint
    # involve(be_born, participant_argument, Person=PER)
    # involve(be_born, attribute_argument, Time=timex2, Place=(GPE, LOC, FAC))
    entities_graph = graph['ACE05/Entities']
    PER = entities_graph['PER']
    ORG = entities_graph['ORG']

    events_graph = graph['ACE05/Events']
    be_born = events_graph['Be-Born']
    participant_argument = events_graph['Participant']
    be_born_participant_person = next(find_event_arg(events_graph, be_born, participant_argument, PER))

    document['index'] = TestSensor(expected_outputs='John works for IBM .')
    dcs['forward'] = TestEdgeSensor('index', to='index', mode='forward', expected_outputs=np.arange(5).reshape(1,5))
    span['emb'] = TestSensor('index', expected_outputs=np.random.randn(5,300))
    span[PER] = TestLearner('index', expected_outputs=np.random.rand(5,2))
    span[ORG] = TestLearner('index', expected_outputs=np.random.rand(5,2))
    span[be_born] = TestLearner('index', expected_outputs=np.random.rand(5,2))
    # 25 pairs in this example
    pair['index'] = CandidateSensor(span['index'], forward=lambda *_: True)
    def pair_emb(span_emb):
        embs = cartesian_concat(span_emb, span_emb)
        # embs = embs.view(-1, embs.shape[-1])
        return embs
    pair['emb'] = FunctionalSensor(span['emb'], forward=pair_emb)
    pair[be_born_participant_person] = TestLearner('emb', expected_outputs=np.random.rand(5, 5, 2))

    program = POIProgram(graph, poi=(
        pair[be_born_participant_person],
        pair['emb'],
        pair['index'],
        span[be_born],
        span[PER],
        span[ORG],
        span['emb'],
        span['index'],
        document['index'],
        ))

    return program
