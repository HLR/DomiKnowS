'''
# Example: Entity-Mention-Relation (EMR)
## Pipeline
This example follows the pipeline we discussed in our preliminary paper.
1. Ontology Declaration
2. Model Declaration
3. Explicit inference
'''
import sys
sys.path.append("../new_interface")
sys.path.append("../..")

from Graphs.Sensors.sentenceSensors import SentenceBertEmbedderSensor, \
    SentenceFlairEmbedderSensor, SentenceGloveEmbedderSensor, FlairSentenceSensor, SentencePosTagger, TensorReaderSensor
from Graphs.Sensors.wordSensors import WordEmbedding, BetweenIndexGenerator, PairIndexGenerator, \
    MultiplyCatSensor, BetweenEncoderSensor, WordPosTaggerSensor, PhraseEntityTagger
from Graphs.Sensors.edgeSensors import FlairSentenceToWord, WordToPhraseTransformer, PhraseToPair, SentenceToWordPos, WordToPhraseTagTransformer
from Graphs.Sensors.relationSensors import RelationReaderSensor, RangeCreatorSensor
from regr.sensor.pytorch.sensors import TorchSensor, ReaderSensor, NominalSensor, ConcatAggregationSensor, ProbabilitySelectionEdgeSensor, \
    MaxAggregationSensor, TorchEdgeSensor, LastAggregationSensor, ConcatSensor, ListConcator, MeanAggregationSensor, FirstAggregationSensor
from regr.sensor.pytorch.learners import LSTMLearner, FullyConnectedLearner, TorchLearner, FullyConnectedLearnerRelu
from data.reader import SimpleReader


def model_declaration():
    from Graphs.graph import graph, rel_phrase_contains_word, rel_sentence_contains_phrase, rel_sentence_contains_word, rel_pair_phrase1, rel_pair_phrase2

    print("model started")
    graph.detach()

    sentence = graph['linguistic/sentence']
    word = graph['linguistic/word']

    FAC = graph['application/FAC']
    GPE = graph['application/GPE']
    PER = graph['application/PER']
    ORG = graph['application/ORG']
    LOC = graph['application/LOC']
    VEH = graph['application/VEH']
    WEA = graph['application/WEA']

    sentence['raw'] = ReaderSensor(keyword='raw')
    sentence['flair_sentence'] = FlairSentenceSensor('raw')
    # sentence['raw'] = Sensor()
    sentence['bert'] = SentenceBertEmbedderSensor('flair_sentence')
    sentence['glove'] = SentenceGloveEmbedderSensor('flair_sentence')
    sentence['flair'] = SentenceFlairEmbedderSensor('flair_sentence')
    sentence['pos'] = SentencePosTagger('flair_sentence')
    sentence['raw_ready'] = TorchSensor('pos', 'bert', 'glove', 'flair', output='flair_sentence')
    rel_sentence_contains_word['forward'] = FlairSentenceToWord('raw_ready', mode="forward", keyword="raw_ready")
    rel_sentence_contains_word['forward'] = SentenceToWordPos('raw_ready', mode="forward", keyword="pos")
    word['embedding'] = WordEmbedding('raw_ready', edges=[rel_sentence_contains_word['forward']])
    postags = ['INTJ', 'AUX', 'CCONJ', 'SYM', 'ADJ', 'VERB', 'PART', 'PROPN', 'PUNCT', 'ADP', 'NOUN', 'PRON', 'ADV', 'DET', 'X', 'NUM']
    word['pos_encode'] = WordPosTaggerSensor('pos', vocab=postags, edges=[rel_sentence_contains_word['forward']])
    word['features'] = ListConcator('embedding', 'pos_encode')
    # word['features'] = TensorReaderSensor(keyword="features")
    word['encode'] = LSTMLearner('features', input_dim=5236, hidden_dim=240, num_layers=1, bidirectional=True)

    # rel_phrase_contains_word['backward'] = BILTransformer('raw_ready', 'boundary', mode="backward")

    # phrase['encode'] = MaxAggregationSensor("raw_ready", edge=rel_phrase_contains_word['backward'], map_key="encode")

    word['encode_final'] = FullyConnectedLearnerRelu('encode', input_dim=480, output_dim=480)

    word[FAC] = ReaderSensor(keyword=FAC.name, label=True)
    word[GPE] = ReaderSensor(keyword=GPE.name, label=True)
    word[PER] = ReaderSensor(keyword=PER.name, label=True)
    word[ORG] = ReaderSensor(keyword=ORG.name, label=True)
    word[LOC] = ReaderSensor(keyword=LOC.name, label=True)
    word[VEH] = ReaderSensor(keyword=VEH.name, label=True)
    word[WEA] = ReaderSensor(keyword=WEA.name, label=True)

    word[FAC] = FullyConnectedLearner('encode_final', input_dim=480, output_dim=2)
    word[GPE] = FullyConnectedLearner('encode_final', input_dim=480, output_dim=2)
    word[PER] = FullyConnectedLearner('encode_final', input_dim=480, output_dim=2)
    word[ORG] = FullyConnectedLearner('encode_final', input_dim=480, output_dim=2)
    word[LOC] = FullyConnectedLearner('encode_final', input_dim=480, output_dim=2)
    word[VEH] = FullyConnectedLearner('encode_final', input_dim=480, output_dim=2)
    word[WEA] = FullyConnectedLearner('encode_final', input_dim=480, output_dim=2)

    from base import ACEGraph, PytorchSolverGraph, NewGraph
    #
    ACEsolver = ACEGraph(PytorchSolverGraph(NewGraph(graph)))
    return ACEsolver


#### The main entrance of the program.
def main():
    # paths = ["ACE_JSON/test/data0.pickle", "ACE_JSON/test/data1.pickle", "ACE_JSON/test/data2.pickle"]
    # paths = ["ACE_JSON/train/data0.pickle", "ACE_JSON/train/data1.pickle", "ACE_JSON/train/data2.pickle", "ACE_JSON/train/data3.pickle", "ACE_JSON/train/data4.pickle", "ACE_JSON/train/data5.pickle", "ACE_JSON/train/data6.pickle", "ACE_JSON/train/data7.pickle", "ACE_JSON/train/data8.pickle", "ACE_JSON/train/data9.pickle", "ACE_JSON/train/data10.pickle"]
    paths = ["ACE_JSON/train/result0-3_1.json"]
    updated_graph = model_declaration()
    # updated_graph.load()
    updated_graph.structured_train_constraint(iterations=50, paths=paths, ratio=1)
    # updated_graph.load()
    # paths = ["ACE_JSON/test/data0.pickle", "ACE_JSON/test/data1.pickle", "ACE_JSON/test/data2.pickle"]
    # updated_graph.predConstraint(paths=paths)
    # updated_graph.PredictionTime(sentence=str(input()))

####
"""
This example show a full pipeline how to work with `regr`.
"""
main()
