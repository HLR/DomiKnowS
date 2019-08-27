from regr.sensor.allennlp.sensor import SentenceSensor, LabelSensor, CartesianProduct3Sensor,ConcatSensor,NGramSensor,CartesianProductSensor,TokenDepDistSensor,TokenDepSensor,TokenDistantSensor,TokenLcaSensor,WordDistantSensor
from regr.sensor.allennlp.learner import SentenceEmbedderLearner, RNNLearner, LogisticRegressionLearner,MLPLearner,ConvLearner


from regr.graph.allennlp import AllenNlpGraph
from SpRL_reader import SpRLSensorReader as Reader
from config import Config
from utils import seed


REGR_DEBUG=1
def ontology_declaration():
    from spGraph import splang_Graph
    return splang_Graph




def model_declaration(graph, config):

    graph.detach()


    sentence = graph['linguistic/sentence']
    word = graph['linguistic/phrase']
    pair = graph['linguistic/pair']


    landmark = graph['application/LANDMARK']
    trajector = graph['application/TRAJECTOR']
    spatialindicator = graph['application/SPATIALINDICATOR']
    none = graph['application/NONE']


    region = graph['application/region']
    relation_none=graph['application/relation_none']
    direction = graph['application/direction']
    distance=graph['application/distance']

    is_triplet=graph['application/is_triplet']
    # is_not_triplet=graph['application/is_not_triplet']

    reader = Reader()

    sentence['raw'] = SentenceSensor(reader, 'sentence')
    word['raw']=SentenceEmbedderLearner('word', config.embedding_dim, sentence['raw'],pretrained_file=config.pretrained_files['word'])
    word['dep']=SentenceEmbedderLearner('dep_tag', config.embedding_dim, sentence['raw'])
    word['pos'] = SentenceEmbedderLearner('pos_tag', config.embedding_dim, sentence['raw'])
    word['lemma'] = SentenceEmbedderLearner('lemma_tag', config.embedding_dim, sentence['raw'])
    word['headword'] = SentenceEmbedderLearner('headword_tag', config.embedding_dim, sentence['raw'])
    word['phrasepos'] = SentenceEmbedderLearner('phrasepos_tag', config.embedding_dim, sentence['raw'])


    word['all'] = ConcatSensor(word['raw'],word['dep'],word['pos'],word['lemma'],word['headword'],word['phrasepos'])
   # word['all'] = ConcatSensor(word['raw'])
    word['ngram'] = NGramSensor(config.ngram, word['all'])
    word['encode'] = RNNLearner(word['ngram'], layers=2, dropout=config.dropout)


    #pair['emb'] = CartesianProduct3Sensor(word['raw'])

    landmark['label'] = LabelSensor(reader,'LANDMARK', output_only=True)
    trajector['label'] = LabelSensor(reader,'TRAJECTOR', output_only=True)
    spatialindicator['label'] = LabelSensor(reader,'SPATIALINDICATOR', output_only=True)
    none['label'] = LabelSensor(reader,'NONE', output_only=True)

    landmark['label'] = LogisticRegressionLearner(word['encode'])
    trajector['label'] = LogisticRegressionLearner(word['encode'])
    spatialindicator['label'] = LogisticRegressionLearner(word['encode'])
    none['label'] = LogisticRegressionLearner(word['encode'])


    word['compact'] = MLPLearner([64,], word['encode'])
    pair['cat'] = CartesianProduct3Sensor(word['compact'])
    #
    #pair['word_dist'] = WordDistantSensor(16, 64, sentence['raw'])
    #pair['tkn_dep'] = TokenDepSensor(sentence['raw'])
    # pair['tkn_dep_dist'] = TokenDepDistSensor(8, 64, sentence['raw'])
    # pair['onehots'] = ConcatSensor(pair['tkn_dist'], pair['tkn_dep'], pair['tkn_dep_dist'])
    #
    #
    #pair['emb'] = MLPLearner([128,], pair['onehots'], activation=None)
    #
    #pair['tkn_lca'] = TokenLcaSensor(sentence['raw'], word['compact'])
    #
    #pair['all'] = ConcatSensor(pair['cat'],  pair['word_dist'])
    #pair['encode'] = ConvLearner([None, None], 5, pair['all'], dropout=config.dropout)
    region['label'] = LabelSensor(reader, 'region', output_only=True)
    relation_none['label'] = LabelSensor(reader, 'relation_none', output_only=True)
    direction['label'] = LabelSensor(reader, 'direction', output_only=True)
    distance['label'] = LabelSensor(reader, 'distance', output_only=True)
    is_triplet['label'] = LabelSensor(reader, 'is_triplet', output_only=True)

    region['label'] = LogisticRegressionLearner(pair['cat'])
    relation_none['label'] = LogisticRegressionLearner(pair['cat'])
    direction['label'] = LogisticRegressionLearner(pair['cat'])
    distance['label'] = LogisticRegressionLearner(pair['cat'])
    is_triplet['label'] = LogisticRegressionLearner(pair['cat'])



    lbp = AllenNlpGraph(graph)
    return lbp


def main():

    graph = ontology_declaration()

    lbp = model_declaration(graph, Config.Model)

    seed()
    lbp.train(Config.Data, Config.Train)
    lbp.save('/tmp/emr')

####
"""
This example show a full pipeline how to work with `regr`.
"""
if __name__ == '__main__':
    main()
