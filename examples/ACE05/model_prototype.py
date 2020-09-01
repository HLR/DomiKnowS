import torch
import numpy as np

from regr.program import POIProgram
from regr.sensor.pytorch.sensors import FunctionalSensor
from regr.sensor.pytorch.query_sensor import CandidateSensor
from test_regr.sensor.pytorch.sensors import TestSensor, TestEdgeSensor

from models import cartesian_concat


def model(graph, ):
    graph.detach()

    ling_graph = graph['linguistic']
    document = ling_graph['document']
    token = ling_graph['token']
    span = ling_graph['span']
    pair = ling_graph['pair']
    dct = document.relate_to(token)[0]
    # sarg1, sarg2 = span.has_a()
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
    dcs['forward'] = TestEdgeSensor('index', to='index', mode='forward', expected_outputs=np.arange(5).reshape(1,5))
    span['emb'] = TestSensor('index', expected_outputs=np.random.randn(5,300))
    # 100 pairs in this example
    pair['index'] = CandidateSensor(span['index'], forward=lambda *_: True)
    def pair_emb(span_emb):
        embs = cartesian_concat(span_emb, span_emb)
        # embs = embs.view(-1, embs.shape[-1])
        return embs
    pair['emb'] = FunctionalSensor(span['emb'], forward=pair_emb)
    pair[be_born_participant_person] = TestSensor('emb', expected_outputs=np.random.rand(100, 2))

    program = POIProgram(graph, poi=(
        # pair[be_born_participant_person],
        pair['emb'],
        pair['index'],
        span['emb'],
        span['index'],
        document['index'],
        # token['index'],
        ))

    return program
