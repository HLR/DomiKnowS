import torch

from regr.program import POIProgram
from regr.sensor.pytorch.sensors import ReaderSensor, ConstantSensor, FunctionalSensor, FunctionalReaderSensor
from regr.sensor.pytorch.learners import ModuleLearner
from regr.sensor.pytorch.query_sensor import CandidateSensor, CandidateRelationSensor

from sensors.tokenizers import TokenizerEdgeSensor
from models import Tokenizer, BERT, token_to_span_candidate, span_candidate_emb, span_label, span_emb, find_is_a


def model(graph):
    graph.detach()

    ling_graph = graph['linguistic']
    ace05_graph = graph['ACE05']
    entities_graph = ace05_graph['Entities']
    relations_graph = ace05_graph['Relations']
    events_graph = ace05_graph['Events']

    document = ling_graph['document']
    token = ling_graph['token']
    span_candidate = ling_graph['span_candidate']
    span = ling_graph['span']
    document_contains_word = document.relate_to(token)[0]
    span_is_span_candidate = span.relate_to(span_candidate)[0]

    document['index'] = ReaderSensor(keyword='text')
    document_contains_word['forward'] = TokenizerEdgeSensor('index', mode='forward', to=('index', 'ids', 'offset'), tokenizer=Tokenizer())
    token['emb'] = ModuleLearner('ids', module=BERT())
    span_candidate['index'] = CandidateSensor(forward=token_to_span_candidate)
    span_candidate['emb'] = FunctionalSensor(token['emb'], span_candidate['index'], forward=span_candidate_emb)

    # span detection
    span_candidate[span] = ModuleLearner('emb', module=torch.nn.Linear(768*2, 2))
    span_candidate[span] = FunctionalReaderSensor('index', token['offset'], keyword='spans', forward=span_label)

    def span_candidate_to_span(spans, span_candidate, _):
        span_candidate
        # filter based on span_candidate.getAttribute('<span>')
        return True
    span['index'] = CandidateRelationSensor(span_candidate[span], relations=(span_is_span_candidate,), forward=span_candidate_to_span)
    span['emb'] = FunctionalSensor(span_candidate['emb'], span['index'], forward=span_emb)

    # span
    for concept in find_is_a(entities_graph, span):
        print(f'Creating learner/reader for span -> {concept}')
        span[concept] = ModuleLearner('emb', module=torch.nn.Linear(768*2, 2))
        # span[concept] = ConstantSensor(data=, label=True)

    # entity major classes
    entity = entities_graph['Entity']
    for concept in find_is_a(entities_graph, entity):
        if '.' in concept.name:  # skip 'Class.', 'Role.', etc.
            continue
        print(f'Creating learner/reader for entity -> {concept}')
        span[concept] = ModuleLearner('emb', module=torch.nn.Linear(768*2, 2))
        # span[concept] = ConstantSensor(data=, label=True)

        # entity sub classes
        for sub_concept in find_is_a(entities_graph, concept):
            if '.' in sub_concept.name:  # skip 'Class.', 'Role.', etc.
                continue
            print(f'Creating learner/reader for {concept} -> {sub_concept}')
            span[sub_concept] = ModuleLearner('emb', module=torch.nn.Linear(768*2, 2))
            # span[sub_concept] = ConstantSensor(data=, label=True)

    # value major classes
    value = entities_graph['value']
    for concept in find_is_a(entities_graph, value):
        print(f'Creating learner/reader for value -> {concept}')
        span[concept] = ModuleLearner('emb', module=torch.nn.Linear(768*2, 2))
        # span[concept] = ConstantSensor(data=, label=True)

        # value sub classes
        for sub_concept in find_is_a(entities_graph, concept):
            print(f'Creating learner/reader for {concept} -> {sub_concept}')
            span[sub_concept] = ModuleLearner('emb', module=torch.nn.Linear(768*2, 2))
            # span[sub_concept] = ConstantSensor(data=, label=True)

    program = POIProgram(graph, poi=(token, span_candidate, span,))

    return program
