from regr.sensor.allennlp.sensor import SentenceSensor, LabelSensor, CartesianProduct3Sensor,ConcatSensor,NGramSensor,CartesianProductSensor,TokenDepDistSensor,TokenDepSensor,TokenDistantSensor,TokenLcaSensor,TripPhraseDistSensor
from regr.sensor.allennlp.learner import SentenceEmbedderLearner, RNNLearner, LogisticRegressionLearner,MLPLearner,ConvLearner,TripletEmbedderLearner


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
    spatialindicator = graph['application/SPATIALINDICATOR']
    none = graph['application/NONE']


    region = graph['application/region']
    relation_none = graph['application/relation_none']
    direction = graph['application/direction']
    distance = graph['application/distance']

    triplet = graph['application/triplet']
    # is_not_triplet=graph['application/is_not_triplet']

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
    spatialindicator['label'] = LabelSensor(reader, 'SPATIALINDICATOR', output_only=True)
    none['label'] = LabelSensor(reader, 'NONE', output_only=True)

    landmark['label'] = LogisticRegressionLearner(phrase['encode'])
    trajector['label'] = LogisticRegressionLearner(phrase['encode'])
    spatialindicator['label'] = LogisticRegressionLearner(phrase['encode'])
    none['label'] = LogisticRegressionLearner(phrase['encode'])

    phrase['compact'] = MLPLearner([config.compact,], phrase['encode'], activation=None)
    triplet['cat'] = CartesianProduct3Sensor(phrase['compact'])
    #triplet['compact_dist'] = TripPhraseDistSensor(phrase['compact'])
    triplet['raw_dist'] = TripPhraseDistSensor(phrase['raw'])
    triplet['pos_dist'] = TripPhraseDistSensor(phrase['pos'])
    triplet['lemma_dist'] =TripPhraseDistSensor(phrase['lemma'])
    triplet['headword_dist'] = TripPhraseDistSensor(phrase['headword'])
    triplet['phrasepos_dist'] = TripPhraseDistSensor(phrase['phrasepos'])
    # new feature example
    triplet['dummy'] = TripletEmbedderLearner('triplet_dummy', config.embedding_dim, sentence['raw'])
    triplet['all'] = ConcatSensor(triplet['cat'],
                                  #triplet['compact_dist'],
                                  triplet['raw_dist'],
                                  triplet['pos_dist'],
                                  triplet['lemma_dist'],
                                  triplet['headword_dist'],
                                  triplet['phrasepos_dist'],
                                  triplet['dummy']
                                 )

    triplet['label'] = LabelSensor(reader, 'is_triplet', output_only=True)
    region['label'] = LabelSensor(reader, 'region', output_only=True)
    relation_none['label'] = LabelSensor(reader, 'relation_none', output_only=True)
    direction['label'] = LabelSensor(reader, 'direction', output_only=True)
    distance['label'] = LabelSensor(reader, 'distance', output_only=True)

    triplet['label'] = LogisticRegressionLearner(triplet['all'])
    region['label'] = LogisticRegressionLearner(triplet['all'])
    relation_none['label'] = LogisticRegressionLearner(triplet['all'])
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
