import torch

from regr.program import POIProgram
from regr.program.primaldualprogram import PrimalDualProgram
from regr.program.model.pytorch import PoiModel
from regr.program.metric import MacroAverageTracker, PRF1Tracker
from regr.program.loss import NBCrossEntropyLoss, NBCrossEntropyIMLoss
from regr.sensor.pytorch.sensors import ReaderSensor, FunctionalSensor, FunctionalReaderSensor, cache, TorchCache, JointSensor
from regr.sensor.pytorch.tokenizers.transformers import TokenizerEdgeSensor
from regr.sensor.pytorch.relation_sensors import EdgeSensor
from regr.sensor.pytorch.learners import ModuleLearner
from regr.sensor.pytorch.relation_sensors import CompositionCandidateSensor

from sensors.readerSensor import MultiLevelReaderSensor, SpanLabelSensor, CustomMultiLevelReaderSensor, LabelConstantSensor
from models import Tokenizer, BERT, SpanClassifier, span_to_pair_emb, find_is_a, find_event_arg, token_to_span_label, makeSpanPairs, makeSpanAnchorPairs


def model(graph):
    from ace05.graph import graph, entities_graph, events_graph
    from ace05.graph import document, token, token_b, token_i, token_o, span, span_annotation, anchor_annotation, event, pair

    # document
    document['text'] = ReaderSensor(keyword='text')

    # document -> token
    document_contains_token = document.relate_to(token)[0]
    token[document_contains_token, 'ids', 'offset'] = cache(JointSensor)(document['text'], forward=Tokenizer(), cache=TorchCache(path="./cache/tokenizer"))
    token['emb'] = ModuleLearner('ids', module=BERT())

    # span annotation
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

    # token -> span and span equality extention
    span_contains_token = span.relate_to(token)[0]
    span_equal_annotation = span.relate_to(span_annotation)[0]
    anchor_equal_annotation = span.relate_to(anchor_annotation)[0]

    def token_to_span_fn(token_offset, sanno_start, sanno_end, aanno_start, aanno_end):
        num_token = token_offset.shape[0]
        token_start = token_offset[:,0]
        token_end = token_offset[:,1]
        spans = []
        # dummy spans
        for start in range(0, num_token, 4):
            spans.append((start, start+2))
        sannos = []
        for start, end in zip(sanno_start, sanno_end):
            start_token = torch.nonzero(torch.logical_and(token_start <= start, start < token_end), as_tuple=False)[0, 0]
            end_token = torch.nonzero(torch.logical_and(token_start < end, end <= token_end), as_tuple=False)[0, 0]
            try:
                span_index = spans.index((start_token, end_token))
            except ValueError:
                span_index = len(spans)
                spans.append((start_token, end_token))
            sannos.append(span_index)
        aannos = []
        for start, end in zip(aanno_start, aanno_end):
            start_token = torch.nonzero(torch.logical_and(token_start <= start, start < token_end), as_tuple=False)[0, 0]
            end_token = torch.nonzero(torch.logical_and(token_start < end, end <= token_end), as_tuple=False)[0, 0]
            try:
                span_index = spans.index((start_token, end_token))
            except ValueError:
                span_index = len(spans)
                spans.append((start_token, end_token))
            aannos.append(span_index)

        token_mapping = torch.zeros(len(spans), num_token)
        for j, (start, end) in enumerate(spans):
            token_mapping[j, start:end] = 1
        sanno_mapping = torch.zeros(len(spans), len(sanno_start))
        for i, index in enumerate(sannos):
            sanno_mapping[index, i] = 1
        aanno_mapping = torch.zeros(len(spans), len(aanno_start))
        for i, index in enumerate(aannos):
            aanno_mapping[index, i] = 1
        return token_mapping, sanno_mapping, aanno_mapping
    span[span_contains_token.reversed, span_equal_annotation.reversed, anchor_equal_annotation.reversed] = JointSensor(token['offset'], span_annotation['start'], span_annotation['end'], anchor_annotation['start'], anchor_annotation['end'], forward=token_to_span_fn)
    span['emb'] = FunctionalSensor(span_contains_token.reversed('emb'), forward=lambda x: x)

    document_contains_span = document.relate_to(span)[0]
    span[document_contains_span] = EdgeSensor(span_contains_token.reversed(document_contains_token, fn=lambda x: x.max(dim=1)[0]), relation=document_contains_span, forward=lambda x: x)

    def token_label(span_token):
        num_token = span_token.shape[1]

        token_b = torch.zeros(num_token)
        idx_b = (span_token - torch.nn.functional.pad(span_token, pad=[1,0])[:, :-1] > 0).nonzero()[:, 1]
        token_b[idx_b] = 1
        token_i = torch.zeros(num_token)
        idx_i = span_token.nonzero()[:, 1]
        token_i[idx_i] = 1
        token_i -= token_b
        token_o = torch.ones(num_token)
        token_o -= token_i +token_b

        return token_b, token_i, token_o

    token[token_b, token_i, token_o] = JointSensor(span[span_contains_token.reversed], forward=token_label, label=True)
    token[token_b] = ModuleLearner('emb', module=torch.nn.Linear(768, 2))
    token[token_i] = ModuleLearner('emb', module=torch.nn.Linear(768, 2))
    token[token_o] = ModuleLearner('emb', module=torch.nn.Linear(768, 2))

    # span -> base types
    for concept in find_is_a(entities_graph, span):
        print(f'Creating learner/reader for span -> {concept}')
        span[concept] = ModuleLearner('emb', module=torch.nn.Linear(768, 2))
        # span_annotation[concept] = LabelConstantSensor(concept=concept.name)
        # span[concept] = ConstantSensor(data=, label=True)

    # entity -> major classes
    entity = entities_graph['Entity']
    for concept in find_is_a(entities_graph, entity):
        if '.' in concept.name:  # skip 'Class.', 'Role.', etc.
            continue
        print(f'Creating learner/reader for entity -> {concept}')
        span[concept] = ModuleLearner('emb', module=torch.nn.Linear(768, 2))
        span_annotation[concept] = LabelConstantSensor('type', concept=concept.name)
        def fn(arg, concept):  # add concept for debug, to remove later
            return arg
        span[concept] = FunctionalSensor(span_equal_annotation.reversed(span_annotation[concept]), concept, forward=fn, label=True)

        # entity -> sub classes
        for sub_concept in find_is_a(entities_graph, concept):
            if '.' in sub_concept.name:  # skip 'Class.', 'Role.', etc.
                continue
            print(f'Creating learner/reader for {concept} -> {sub_concept}')
            span[sub_concept] = ModuleLearner('emb', module=torch.nn.Linear(768, 2))
            span_annotation[sub_concept] = LabelConstantSensor('subtype', concept=sub_concept.name)
            # span[sub_concept] = ConstantSensor(data=, label=True)

    # value -> major classes
    value = entities_graph['value']
    for concept in find_is_a(entities_graph, value):
        print(f'Creating learner/reader for value -> {concept}')
        span[concept] = ModuleLearner('emb', module=torch.nn.Linear(768, 2))
        span_annotation[concept] = LabelConstantSensor('type', concept=concept.name)
        # span[concept] = ConstantSensor(data=, label=True)

        # value -> sub classes
        for sub_concept in find_is_a(entities_graph, concept):
            print(f'Creating learner/reader for {concept} -> {sub_concept}')
            span[sub_concept] = ModuleLearner('emb', module=torch.nn.Linear(768, 2))
            span_annotation[sub_concept] = LabelConstantSensor('subtype', concept=sub_concept.name)
            # span[sub_concept] = ConstantSensor(data=, label=True)

    # span -> pair
    arg1, arg2 = pair.relate_to(span)
    pair[arg1.reversed, arg2.reversed] = CompositionCandidateSensor(
        span['emb'],
        relations=(arg1.reversed, arg2.reversed),
        forward=lambda *_, **__: True)
    pair['emb'] = FunctionalSensor(span['emb'], forward=span_to_pair_emb)

    # event
    for concept in find_is_a(events_graph, event):
        print(f'Creating learner/reader for event -> {concept}')
        span[concept] = ModuleLearner('emb', module=torch.nn.Linear(768, 2))
        # span_annotation[concept] = LabelConstantSensor('type', concept=concept.name)
        # span[concept] = ConstantSensor(data=, label=True)

        # event sub classes
        for sub_concept in find_is_a(events_graph, concept):
            print(f'Creating learner/reader for {concept} -> {sub_concept}')
            span[sub_concept] = ModuleLearner('emb', module=torch.nn.Linear(768, 2))
            # span_annotation[sub_concept] = LabelConstantSensor('subtype', concept=sub_concept.name)
            # span[sub_concept] = ConstantSensor(data=, label=True)

            # all event argument rules are associated with event subtypes
            # pair -> event argument
            for event_arg in find_event_arg(events_graph, sub_concept):
                print(f'Creating learner/reader for pair -> {sub_concept.name}\'s {event_arg.name}')
                pair[event_arg] = ModuleLearner('emb', module=torch.nn.Linear(768*2, 2))
                # pair[event_arg] = ?

    # program = POIProgram(graph, poi=(span, pair))
    # program = PrimalDualProgram(graph, PoiModel, poi=(document, token, span, pair))
    program = POIProgram(
        graph,
        poi=(document, token, span, pair),
        loss=MacroAverageTracker(NBCrossEntropyLoss()),
        metric=PRF1Tracker())
    # program = IMLProgram(
    #     graph,
    #     poi=(document, token, span, pair),
    #     loss=MacroAverageTracker(NBCrossEntropyIMLoss(lmbd=0.5)),
    #     metric=PRF1Tracker())

    return program
