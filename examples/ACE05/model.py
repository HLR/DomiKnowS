import torch

from regr.program import POIProgram
from regr.sensor.pytorch.sensors import ReaderSensor, ConstantSensor, FunctionalSensor, FunctionalReaderSensor, TorchEdgeSensor
from regr.sensor.pytorch.learners import ModuleLearner
from regr.sensor.pytorch.relation_sensors import CandidateSensor, CandidateRelationSensor, CandidateEqualSensor

from sensors.tokenizers import TokenizerEdgeSensor
from sensors.readerSensor import MultiLevelReaderSensor, SpanLabelSensor, CustomMultiLevelReaderSensor, LabelConstantSensor
from models import Tokenizer, BERT, SpanClassifier, token_to_span_candidate_emb, span_to_pair_emb, find_is_a, find_event_arg, token_to_span_label, makeSpanPairs, makeSpanAnchorPairs


def model(graph):
    from ace05.graph import graph, entities_graph, events_graph
    from ace05.graph import document, token, span_candidate, span, span_annotation, anchor_annotation, event, pair

    # document
    document['index'] = ReaderSensor(keyword='text')

    # document -> token
    document_contains_token = document.relate_to(token)[0]
    document_contains_token['forward'] = TokenizerEdgeSensor('index', mode='forward', to=('index', 'ids', 'offset'), tokenizer=Tokenizer())
    token['emb'] = ModuleLearner('ids', module=BERT())

    # token -> span
    span_candidate['index'] = CandidateSensor(token['index'], forward=lambda *_: True)
    span_candidate['emb'] = FunctionalSensor(token['emb'], forward=token_to_span_candidate_emb)
    span_candidate['label'] = ModuleLearner('emb', module=SpanClassifier(token_emb_dim=768))

    span_contains_token = span.relate_to(token)[0]
    span_contains_token['backward'] = TorchEdgeSensor(
        span_candidate['label'], to='index', forward=token_to_span_label, mode='backward',)

    span['emb'] = FunctionalSensor(span_candidate['emb'], forward=lambda x: x)

    # span equality extention
    span_annotation['index'] = MultiLevelReaderSensor(keyword="spans.*.mentions.*.head.text")
    span_annotation['start'] = MultiLevelReaderSensor(keyword="spans.*.mentions.*.head.start")
    span_annotation['end'] = MultiLevelReaderSensor(keyword="spans.*.mentions.*.head.end")
    span_annotation['type'] = CustomMultiLevelReaderSensor(keyword="spans.*.type")
    span_annotation['subtype'] = CustomMultiLevelReaderSensor(keyword="spans.*.subtype")
    
    anchor_annotation['index'] = MultiLevelReaderSensor(keyword="events.*.mentions.*.anchor.text")
    anchor_annotation['start'] = MultiLevelReaderSensor(keyword="events.*.mentions.*.anchor.start")
    anchor_annotation['end'] = MultiLevelReaderSensor(keyword="events.*.mentions.*.anchor.end")
    anchor_annotation['type'] = CustomMultiLevelReaderSensor(keyword="events.*.type")
    anchor_annotation['subtype'] = CustomMultiLevelReaderSensor(keyword="events.*.subtype")

    span_equal_annotation = span.relate_to(span_annotation)[0]
    span['match'] = CandidateEqualSensor('index', span_annotation['index'],span_annotation['start'], span_annotation['end'], forward=makeSpanPairs, relations=[span_equal_annotation])
    anchor_equal_annotation = span.relate_to(anchor_annotation)[0]
    span['match1'] = CandidateEqualSensor('index', anchor_annotation['index'], anchor_annotation['start'], anchor_annotation['end'], forward=makeSpanAnchorPairs, relations=[anchor_equal_annotation])
    span['label'] = SpanLabelSensor('match', label=True, concept=span_annotation.name)

    # span -> base types
    for concept in find_is_a(entities_graph, span):
        print(f'Creating learner/reader for span -> {concept}')
        span[concept] = ModuleLearner('emb', module=torch.nn.Linear(768*2, 2))
        # span_annotation[concept] = LabelConstantSensor(concept=concept.name)
        # span[concept] = ConstantSensor(data=, label=True)

    # entity -> major classes
    entity = entities_graph['Entity']
    for concept in find_is_a(entities_graph, entity):
        if '.' in concept.name:  # skip 'Class.', 'Role.', etc.
            continue
        print(f'Creating learner/reader for entity -> {concept}')
        span[concept] = ModuleLearner('emb', module=torch.nn.Linear(768*2, 2))
        span_annotation[concept] = LabelConstantSensor('type', concept=concept.name)
        # span[concept] = ConstantSensor(data=, label=True)

        # entity -> sub classes
        for sub_concept in find_is_a(entities_graph, concept):
            if '.' in sub_concept.name:  # skip 'Class.', 'Role.', etc.
                continue
            print(f'Creating learner/reader for {concept} -> {sub_concept}')
            span[sub_concept] = ModuleLearner('emb', module=torch.nn.Linear(768*2, 2))
            span_annotation[sub_concept] = LabelConstantSensor('subtype', concept=sub_concept.name)
            # span[sub_concept] = ConstantSensor(data=, label=True)

    # value -> major classes
    value = entities_graph['value']
    for concept in find_is_a(entities_graph, value):
        print(f'Creating learner/reader for value -> {concept}')
        span[concept] = ModuleLearner('emb', module=torch.nn.Linear(768*2, 2))
        span_annotation[concept] = LabelConstantSensor('type', concept=concept.name)
        # span[concept] = ConstantSensor(data=, label=True)

        # value -> sub classes
        for sub_concept in find_is_a(entities_graph, concept):
            print(f'Creating learner/reader for {concept} -> {sub_concept}')
            span[sub_concept] = ModuleLearner('emb', module=torch.nn.Linear(768*2, 2))
            span_annotation[sub_concept] = LabelConstantSensor('subtype', concept=sub_concept.name)
            # span[sub_concept] = ConstantSensor(data=, label=True)

    # span -> pair
    pair['index'] = CandidateSensor(span['index'], forward=lambda *_: True)
    pair['emb'] = FunctionalSensor(span['emb'], forward=span_to_pair_emb)

    # event
    for concept in find_is_a(events_graph, event):
        print(f'Creating learner/reader for event -> {concept}')
        span[concept] = ModuleLearner('emb', module=torch.nn.Linear(768*2, 2))
        # span_annotation[concept] = LabelConstantSensor('type', concept=concept.name)
        # span[concept] = ConstantSensor(data=, label=True)

        # event sub classes
        for sub_concept in find_is_a(events_graph, concept):
            print(f'Creating learner/reader for {concept} -> {sub_concept}')
            span[sub_concept] = ModuleLearner('emb', module=torch.nn.Linear(768*2, 2))
            # span_annotation[sub_concept] = LabelConstantSensor('subtype', concept=sub_concept.name)
            # span[sub_concept] = ConstantSensor(data=, label=True)

            # all event argument rules are associated with event subtypes
            # pair -> event argument
            for event_arg in find_event_arg(events_graph, sub_concept):
                print(f'Creating learner/reader for pair -> {sub_concept.name}\'s {event_arg.name}')
                pair[event_arg] = ModuleLearner('emb', module=torch.nn.Linear(768*4, 2))
                # pair[event_arg] = ?

    program = POIProgram(graph, poi=(span, pair))

    return program
