from regr.sensor.allennlp.sensor import SequenceSensor, TokenInSequenceSensor, LabelSensor, CartesianProductSensor
from regr.sensor.allennlp.learner import W2VLearner, RNNLearner, LogisticRegressionLearner

from regr.graph.allennlp import AllenNlpGraph
from SpRL_reader import SpRLSensorReader as Reader
from config import Config
from utils import seed

def ontology_declaration():
    from spGraph import splang_Graph
    return splang_Graph


def model_declaration(graph, config):

    graph.detach()


    sentence = graph['linguistic/sentence']
    phrase = graph['linguistic/phrase']
    pair = graph['linguistic/pair']


    landmark = graph['application/LANDMARK']
    trajector = graph['application/TRAJECTOR']
    spatialindicator = graph['application/SPATIALINDICATOR']
    none = graph['application/NONE']


    region = graph['application/region']

    reader = Reader()

    sentence['raw'] = SequenceSensor(reader, 'sentence')
    phrase['raw'] = TokenInSequenceSensor(sentence['raw'])
    phrase['w2v'] = W2VLearner(config.embedding_dim, phrase['raw'])
    phrase['emb'] = RNNLearner(config.embedding_dim, phrase['w2v'])
    pair['emb'] = CartesianProductSensor(phrase['emb'])

    landmark['label'] = LabelSensor(reader,'LANDMARK', output_only=True)
    trajector['label'] = LabelSensor(reader,'TRAJECTOR', output_only=True)
    spatialindicator['label'] = LabelSensor(reader,'SPATIALINDICATOR', output_only=True)
    none['label'] = LabelSensor(reader,'NONE', output_only=True)


    landmark['label'] = LogisticRegressionLearner(config.embedding_dim * 2, phrase['emb'])
    trajector['label'] = LogisticRegressionLearner(config.embedding_dim * 2, phrase['emb'])
    spatialindicator['label'] = LogisticRegressionLearner(config.embedding_dim * 2, phrase['emb'])
    none['label'] = LogisticRegressionLearner(config.embedding_dim * 2, phrase['emb'])

    region['label'] = LabelSensor(reader, 'region', output_only=True)

    region['label'] = LogisticRegressionLearner(config.embedding_dim * 4, pair['emb'])


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
