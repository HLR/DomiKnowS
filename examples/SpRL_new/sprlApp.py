from regr.sensor.allennlp.sensor import SentenceSensor, LabelSensor, CartesianProduct3Sensor
from regr.sensor.allennlp.learner import SentenceEmbedderLearner, RNNLearner, LogisticRegressionLearner

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

    sentence['raw'] = SentenceSensor(reader, 'sentence')
    phrase['w2v'] = SentenceEmbedderLearner('word', config.embedding_dim, sentence['raw'])
    phrase['emb'] = RNNLearner(phrase['w2v'])
    pair['emb'] = CartesianProduct3Sensor(phrase['emb'])

    landmark['label'] = LabelSensor(reader,'LANDMARK', output_only=True)
    trajector['label'] = LabelSensor(reader,'TRAJECTOR', output_only=True)
    spatialindicator['label'] = LabelSensor(reader,'SPATIALINDICATOR', output_only=True)
    none['label'] = LabelSensor(reader,'NONE', output_only=True)


    landmark['label'] = LogisticRegressionLearner(phrase['emb'])
    trajector['label'] = LogisticRegressionLearner(phrase['emb'])
    spatialindicator['label'] = LogisticRegressionLearner(phrase['emb'])
    none['label'] = LogisticRegressionLearner(phrase['emb'])

    region['label'] = LabelSensor(reader, 'region', output_only=True)

    region['label'] = LogisticRegressionLearner(pair['emb'])


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
