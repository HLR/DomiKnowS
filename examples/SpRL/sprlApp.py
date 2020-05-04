from regr.sensor.allennlp.sensor import SentenceSensor, TensorReaderSensor, LabelSensor, CartesianProduct3Sensor, SelfCartesianProduct3Sensor, ConcatSensor, NGramSensor, CartesianProductSensor, TokenDepDistSensor, TokenDepSensor, TokenDistantSensor, TokenLcaSensor, TripPhraseDistSensor, LabelMaskSensor, JointCandidateSensor, CandidateReaderSensor, ActivationSensor
from regr.sensor.allennlp.learner import SentenceEmbedderLearner, RNNLearner, LogisticRegressionLearner, MLPLearner, ConvLearner, TripletEmbedderLearner


from regr.graph.allennlp import AllenNlpGraph
from SpRL_reader import SpRLSensorReader as Reader
#from SpRL_reader import PickleReader as Reader
from config import Config
from utils import seed


def ontology_declaration():
    from spGraph import splang_Graph
    return splang_Graph


def model_declaration(graph, config):
    graph.detach()

    sentence = graph['linguistic/sentence']
    phrase = graph['linguistic/phrase']

    landmark = graph['application/LANDMARK']
    trajector = graph['application/TRAJECTOR']
    spatial_indicator = graph['application/SPATIAL_INDICATOR']
    #none_entity = graph['application/NONE_ENTITY']

    triplet = graph['application/triplet']
    spatial_triplet = graph['application/spatial_triplet']
    #none_relation = graph['application/none_relation']

    region = graph['application/region']
    direction = graph['application/direction']
    distance = graph['application/distance']

    reader = Reader()

    sentence['raw'] = SentenceSensor(reader, 'sentence')
    phrase['raw'] = SentenceEmbedderLearner('word', config.embedding_dim, sentence['raw'])
    phrase['vec'] = TensorReaderSensor(reader, 'vec', dims=(300,))
    phrase['dep'] = SentenceEmbedderLearner('dep_tag', config.embedding_dim, sentence['raw'])
    phrase['pos'] = SentenceEmbedderLearner('pos_tag', config.embedding_dim, sentence['raw'])
    phrase['lemma'] = SentenceEmbedderLearner('lemma_tag', config.embedding_dim, sentence['raw'])
    phrase['headword'] = SentenceEmbedderLearner('headword_tag', config.embedding_dim, sentence['raw'])
    phrase['phrasepos'] = SentenceEmbedderLearner('phrasepos_tag', config.embedding_dim, sentence['raw'])

    phrase['all'] = ConcatSensor(phrase['raw'], phrase['vec'], phrase['dep'], phrase['pos'], phrase['lemma'], phrase['headword'], phrase['phrasepos'])
    phrase['ngram'] = NGramSensor(config.ngram, phrase['all'])
    phrase['encode'] = RNNLearner(phrase['ngram'], layers=2, dropout=config.dropout)
    phrase['candidate'] = CandidateReaderSensor(reader, 'entity_mask', output_only=True)

    landmark['label'] = LabelSensor(reader, 'LANDMARK', output_only=True)
    trajector['label'] = LabelSensor(reader, 'TRAJECTOR', output_only=True)
    spatial_indicator['label'] = LabelSensor(reader, 'SPATIALINDICATOR', output_only=True)
    #none_entity['label'] = LabelSensor(reader, 'NONE', output_only=True)

    landmark['label'] = LogisticRegressionLearner(phrase['encode'])
    trajector['label'] = LogisticRegressionLearner(phrase['encode'])
    spatial_indicator['label'] = LogisticRegressionLearner(phrase['encode'])
    #none_entity['label'] = LogisticRegressionLearner(phrase['encode'])

    phrase['compact'] = MLPLearner([config.compact,], phrase['encode'], activation=None)
    triplet['cat'] = SelfCartesianProduct3Sensor(phrase['compact'])
    # new feature example
    triplet['tr_1'] = TripletEmbedderLearner('triplet_feature1', config.embedding_dim, sentence['raw'])
    triplet['tr_2'] = TripletEmbedderLearner('triplet_feature2', config.embedding_dim, sentence['raw'])
    triplet['tr_3'] = TripletEmbedderLearner('triplet_feature3', config.embedding_dim, sentence['raw'])
    triplet['tr_4'] = TripletEmbedderLearner('triplet_feature4', config.embedding_dim, sentence['raw'])
    triplet['tr_5'] = TripletEmbedderLearner('triplet_feature5', config.embedding_dim, sentence['raw'])
    triplet['tr_6'] = TripletEmbedderLearner('triplet_feature6', config.embedding_dim, sentence['raw'])
    triplet['tr_7'] = TripletEmbedderLearner('triplet_feature7', config.embedding_dim, sentence['raw'])
    triplet['tr_8'] = TripletEmbedderLearner('triplet_feature8', config.embedding_dim, sentence['raw'])
    triplet['encode'] = ConcatSensor(triplet['cat'],
                                    triplet['tr_1'],
                                    triplet['tr_2'],
                                    triplet['tr_3'],
                                    triplet['tr_4'],
                                    triplet['tr_5'],
                                    triplet['tr_6'],
                                    triplet['tr_7'],
                                    triplet['tr_8']
                                    )
    # triplet['encode'] = MLPLearner([config.compact, ], triplet['all'])

    triplet['candidate'] = CandidateReaderSensor(reader, 'triplet_mask', output_only=True)

    spatial_triplet['label'] = LabelSensor(reader, 'is_triplet', output_only=True)
    #none_relation['label'] = LabelSensor(reader, 'relation_none', output_only=True)

    region['label'] = LabelSensor(reader, 'region', output_only=True)
    direction['label'] = LabelSensor(reader, 'direction', output_only=True)
    distance['label'] = LabelSensor(reader, 'distance', output_only=True)

    spatial_triplet['label'] = LogisticRegressionLearner(triplet['encode'])
    #none_relation['label'] = LogisticRegressionLearner(triplet['encode'])

    region['label'] = LogisticRegressionLearner(triplet['encode'])
    direction['label'] = LogisticRegressionLearner(triplet['encode'])
    distance['label'] = LogisticRegressionLearner(triplet['encode'])

    lbp = AllenNlpGraph(graph, **config.graph, post_action=log_output(config.log_dir))
    return lbp

def log_output(log_dir):
    from collections import defaultdict
    import re
    import os
    import json

    NAME_PATTEN_RAW = re.compile(r'^spLanguage\/linguistic\/sentence\/raw\/.*$')
    NAME_PATTEN_EN = re.compile(r'^spLanguage\/application\/([A-Z]+\w+)\/label\/.+learner-?\d*$')
    NAME_PATTEN_TR = re.compile(r'^spLanguage\/application\/([a-z]+\w+)\/label\/.+learner-?\d*$')
    NAME_PATTEN_EN_CANDIDATE=re.compile(r'^spLanguage\/linguistic\/phrase\/candidate\/.*$')
    NAME_PATTEN_TR_CANDIDATE=re.compile(r'^spLanguage\/application\/triplet\/candidate\/.*$')

    def log(data, data_type=''):
        dataset_type = data['dataset_type']
        epoch_num = data['epoch_num']
        raw = None
        for name, value in data.items():
            match = NAME_PATTEN_RAW.match(name)
            if match:
                raw = value
                break

        entity_candidate = None
        triplet_candidate = None
        for name, value in data.items():
            match = NAME_PATTEN_EN_CANDIDATE.match(name)
            if match:
                assert entity_candidate is None, 'Should contain AT MOST one entity candidate. Multiple are detected.'
                entity_candidate = value
                continue
            match = NAME_PATTEN_TR_CANDIDATE.match(name)
            if match:
                assert triplet_candidate is None, 'Should contain AT MOST one triplet candidate. Multiple are detected.'
                triplet_candidate = value
                continue

        entities = {}
        triplets = {}
        for name, value in data.items():
            match = NAME_PATTEN_EN.match(name)
            if match:
                entities[match[1]] = value
                continue
            match = NAME_PATTEN_TR.match(name)
            if match:
                triplets[match[1]] = value
                continue

        for i, phrases in enumerate(raw):
            out = defaultdict(dict)
            df = phrases[0].doc
            out['id'] = df.metas['id']
            out['docno'] = df.metas['docno']
            out['image'] = df.metas['image']
            out['start'] = df.metas['start']
            out['end'] = df.metas['end']
            out['text'] = df.sentence
            for name, entity in entities.items():
                _, entity_idx = entity[i].max(dim=-1)
                if entity_candidate is not None:
                    entity_idx = entity_idx * entity_candidate[i]
                for idx in entity_idx.nonzero():
                    phrase = phrases[idx]
                    out[name][id(phrase)] = (phrase.start, phrase.end)
            for name, triplet in triplets.items():
                _, triplet_idx = triplet[i].max(dim=-1)
                if triplet_candidate is not None:
                    triplet_idx = triplet_idx * triplet_candidate[i].long()
                for idx in triplet_idx.nonzero():
                    lm, tr, sp = idx
                    lm = phrases[lm]
                    tr = phrases[tr]
                    sp = phrases[sp]
                    if 0 not in out['LANDMARK'] and lm.is_dummy_:
                        out['LANDMARK'][0] = (-1, -1)
                    if 0 not in out['TRAJECTOR'] and tr.is_dummy_:
                        out['TRAJECTOR'][0] = (-1, -1)
                    if 0 not in out['SPATIAL_INDICATOR'] and sp.is_dummy_:
                        out['SPATIAL_INDICATOR'][0] = (-1, -1)
                    lmid = id(lm) if not lm.is_dummy_ else 0
                    trid = id(tr) if not tr.is_dummy_ else 0
                    spid = id(sp) if not sp.is_dummy_ else 0
                    out[name][id((lm, tr, sp))] = (lmid, trid, spid)
            fout_path = os.path.join(log_dir, 'out-{}-{}-{}.jsonl'.format(dataset_type[i], str(epoch_num[i]), data_type))
            with open(fout_path, 'a') as fout:
                fout.write(json.dumps(out) + '\n')
    return log

def main():
    graph = ontology_declaration()

    lbp = model_declaration(graph, Config.Model)

    seed()
    lbp.train(Config.Data, Config.Train)
    save_to = Config.Train.trainer.serialization_dir or '/tmp/emr'
    lbp.save(save_to, config=Config)


####
"""
This example show a full pipeline how to work with `regr`.
"""
if __name__ == '__main__':
    main()
