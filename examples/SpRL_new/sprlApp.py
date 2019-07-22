from regr.sensor.allennlp.sensor import SentenceSensor, LabelSensor, CartesianProduct3Sensor,ConcatSensor,NGramSensor
from regr.sensor.allennlp.learner import SentenceEmbedderLearner, RNNLearner, LogisticRegressionLearner


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

    reader = Reader()

    sentence['raw'] = SentenceSensor(reader, 'sentence')
    word['raw']=SentenceEmbedderLearner('word', config.embedding_dim, sentence['raw'],pretrained_file=config.pretrained_files['word'])
    word['dep']=SentenceEmbedderLearner('dep_tag', config.embedding_dim, sentence['raw'])
    word['pos'] = SentenceEmbedderLearner('pos_tag', config.embedding_dim, sentence['raw'])

   # phrase['emb'] = RNNLearner(phrase['w2v'])

    word['all'] = ConcatSensor(word['raw'],word['dep'],word['pos'])

    word['ngram'] = NGramSensor(config.ngram, word['all'])
    word['encode'] = RNNLearner(word['ngram'], layers=2, dropout=config.dropout)

#    pair['emb'] = CartesianProduct3Sensor(phrase['emb'])

    landmark['label'] = LabelSensor(reader,'LANDMARK', output_only=True)
    trajector['label'] = LabelSensor(reader,'TRAJECTOR', output_only=True)
    spatialindicator['label'] = LabelSensor(reader,'SPATIALINDICATOR', output_only=True)
    none['label'] = LabelSensor(reader,'NONE', output_only=True)


    landmark['label'] = LogisticRegressionLearner(word['encode'])
    trajector['label'] = LogisticRegressionLearner(word['encode'])
    spatialindicator['label'] = LogisticRegressionLearner(word['encode'])
    none['label'] = LogisticRegressionLearner(word['encode'])

    # region['label'] = LabelSensor(reader, 'region', output_only=True)
    # relation_none['label'] = LabelSensor(reader, 'relation_none', output_only=True)
    # direction['label'] = LabelSensor(reader, 'direction', output_only=True)
    # distance['label'] = LabelSensor(reader, 'distance', output_only=True)


    #region['label'] = LogisticRegressionLearner(pair['encode'])
    # relation_none['label'] = LogisticRegressionLearner(pair['encode'])
    # direction['label'] = LogisticRegressionLearner(pair['encode'])
    # distance['label'] = LogisticRegressionLearner(pair['encode'])

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
