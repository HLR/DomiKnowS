from regr.sensor.allennlp.sensor import SentenceSensor, LabelSensor, CartesianProduct3Sensor, ConcatSensor, NGramSensor, CartesianProductSensor, TokenDepDistSensor, TokenDepSensor, TokenDistantSensor, TokenLcaSensor, TripPhraseDistSensor, LabelMaskSensor, JointCandidateSensor
from regr.sensor.allennlp.learner import SentenceEmbedderLearner, RNNLearner, LogisticRegressionLearner, MLPLearner, ConvLearner, TripletEmbedderLearner


from regr.graph.allennlp import AllenNlpGraph
#from SpRL_reader import SpRLSensorReader as Reader
from reader_new import SpRLSensorReader as Reader
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
    none_entity = graph['application/NONE_ENTITY']

    triplet = graph['application/triplet']
    spatial_triplet = graph['application/spatial_triplet']
    none_relation = graph['application/none_relation']

    region = graph['application/region']
    direction = graph['application/direction']
    distance = graph['application/distance']

    reader = Reader()

    sentence['raw'] = SentenceSensor(reader, 'sentence')
    phrase['raw'] = SentenceEmbedderLearner('word', config.embedding_dim, sentence['raw'])
    phrase['dep'] = SentenceEmbedderLearner('dep_tag', config.embedding_dim, sentence['raw'])
    phrase['pos'] = SentenceEmbedderLearner('pos_tag', config.embedding_dim, sentence['raw'])
    phrase['lemma'] = SentenceEmbedderLearner('lemma_tag', config.embedding_dim, sentence['raw'])
    phrase['headword'] = SentenceEmbedderLearner('headword_tag', config.embedding_dim, sentence['raw'])
    phrase['phrasepos'] = SentenceEmbedderLearner('phrasepos_tag', config.embedding_dim, sentence['raw'])

    phrase['all'] = ConcatSensor(phrase['raw'], phrase['dep'], phrase['pos'], phrase['lemma'], phrase['headword'], phrase['phrasepos'])
    phrase['ngram'] = NGramSensor(config.ngram, phrase['all'])
    phrase['encode'] = RNNLearner(phrase['ngram'], layers=2, dropout=config.dropout)

    landmark['label'] = LabelSensor(reader, 'LANDMARK', output_only=True)
    trajector['label'] = LabelSensor(reader, 'TRAJECTOR', output_only=True)
    spatial_indicator['label'] = LabelSensor(reader, 'SPATIALINDICATOR', output_only=True)
    none_entity['label'] = LabelSensor(reader, 'NONE', output_only=True)

    landmark['label'] = LogisticRegressionLearner(phrase['encode'])
    trajector['label'] = LogisticRegressionLearner(phrase['encode'])
    spatial_indicator['label'] = LogisticRegressionLearner(phrase['encode'])
    none_entity['label'] = LogisticRegressionLearner(phrase['encode'])

    phrase['compact'] = MLPLearner([config.compact,], phrase['encode'], activation=None)
    triplet['cat'] = CartesianProduct3Sensor(phrase['compact'])
    #triplet['compact_dist'] = TripPhraseDistSensor(phrase['compact'])
    triplet['raw_dist'] = TripPhraseDistSensor(phrase['raw'])
    triplet['pos_dist'] = TripPhraseDistSensor(phrase['pos'])
    triplet['lemma_dist'] =TripPhraseDistSensor(phrase['lemma'])
    triplet['headword_dist'] = TripPhraseDistSensor(phrase['headword'])
    triplet['phrasepos_dist'] = TripPhraseDistSensor(phrase['phrasepos'])
    triplet['dependency_dist'] = TripPhraseDistSensor(phrase['dep'])
    # new feature example
    triplet['tr_1'] = TripletEmbedderLearner('triplet_feature1', config.embedding_dim, sentence['raw'])
    triplet['tr_2'] = TripletEmbedderLearner('triplet_feature2', config.embedding_dim, sentence['raw'])
    triplet['tr_3'] = TripletEmbedderLearner('triplet_feature3', config.embedding_dim, sentence['raw'])
    triplet['tr_4'] = TripletEmbedderLearner('triplet_feature4', config.embedding_dim, sentence['raw'])
    triplet['tr_5'] = TripletEmbedderLearner('triplet_feature5', config.embedding_dim, sentence['raw'])
    triplet['all'] = ConcatSensor(triplet['cat'],
                                  #triplet['compact_dist'],
                                  triplet['raw_dist'],
                                  triplet['pos_dist'],
                                  triplet['lemma_dist'],
                                  triplet['headword_dist'],
                                  triplet['phrasepos_dist'],
                                  triplet['dependency_dist'],
                                  triplet['tr_1'],
                                  triplet['tr_2'],
                                  triplet['tr_3'],
                                  triplet['tr_4'],
                                  triplet['tr_5']
                                 )

    triplet['label_mask'] = LabelMaskSensor(reader, 'triplet_mask', output_only=True)
    spatial_triplet['candidate'] = JointCandidateSensor(landmark['label'], trajector['label'], spatial_indicator['label'])

    spatial_triplet['label'] = LabelSensor(reader, 'is_triplet', output_only=True)
    none_relation['label'] = LabelSensor(reader, 'relation_none', output_only=True)

    region['label'] = LabelSensor(reader, 'region', output_only=True)
    direction['label'] = LabelSensor(reader, 'direction', output_only=True)
    distance['label'] = LabelSensor(reader, 'distance', output_only=True)

    spatial_triplet['label'] = LogisticRegressionLearner(triplet['all'])
    none_relation['label'] = LogisticRegressionLearner(triplet['all'])

    region['label'] = LogisticRegressionLearner(triplet['all'])
    direction['label'] = LogisticRegressionLearner(triplet['all'])
    distance['label'] = LogisticRegressionLearner(triplet['all'])

    lbp = AllenNlpGraph(graph, **config.graph)
    return lbp


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
