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
    SentenceFlairEmbedderSensor, SentenceGloveEmbedderSensor, FlairSentenceSensor, SentencePosTagger
from Graphs.Sensors.wordSensors import WordEmbedding, BetweenIndexGenerator, PairIndexGenerator, \
    MultiplyCatSensor, BetweenEncoderSensor, WordPosTaggerSensor
from Graphs.Sensors.edgeSensors import FlairSentenceToWord, WordToPhraseTransformer, PhraseToPair, SentenceToWordPos
from regr.sensor.pytorch.sensors import TorchSensor, ReaderSensor, NominalSensor, ConcatAggregationSensor, ProbabilitySelectionEdgeSensor, \
    MaxAggregationSensor, TorchEdgeSensor, LastAggregationSensor, ConcatSensor, ListConcator
from regr.sensor.pytorch.learners import LSTMLearner, FullyConnectedLearner, TorchLearner
from data.reader import SimpleReader


def model_declaration():
    from Graphs.graph import graph, rel_phrase_contains_word, rel_sentence_contains_phrase, rel_sentence_contains_word, rel_pair_phrase1, rel_pair_phrase2

    print("model started")
    graph.detach()

    sentence = graph['linguistic/sentence']
    word = graph['linguistic/word']
    phrase = graph['linguistic/phrase']
    pair = graph['linguistic/pair']

    FAC = graph['application/FAC']
    GPE = graph['application/GPE']
    PER = graph['application/PER']
    ORG = graph['application/ORG']
    LOC = graph['application/LOC']
    VEH = graph['application/VEH']
    WEA = graph['application/WEA']
    ART = graph['application/ART']

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
    word['encode'] = LSTMLearner('features', input_dim=5236, hidden_dim=240, num_layers=1, bidirectional=True)
    # word['boundary'] = FullyConnectedLearner('encode', input_dim=480, output_dim=4)
    # word['boundary'] = ReaderSensor(keyword='boundary')

    # rel_phrase_contains_word['backward'] = BILTransformer('raw_ready', 'boundary', mode="backward")

    # phrase['encode'] = MaxAggregationSensor("raw_ready", edge=rel_phrase_contains_word['backward'], map_key="encode")

    word[FAC] = ReaderSensor(keyword=FAC.name, label=True)
    word[GPE] = ReaderSensor(keyword=GPE.name, label=True)
    word[PER] = ReaderSensor(keyword=PER.name, label=True)
    word[ORG] = ReaderSensor(keyword=ORG.name, label=True)
    word[LOC] = ReaderSensor(keyword=LOC.name, label=True)
    word[VEH] = ReaderSensor(keyword=VEH.name, label=True)
    word[WEA] = ReaderSensor(keyword=WEA.name, label=True)

    word[FAC] = FullyConnectedLearner('encode', input_dim=480, output_dim=2)
    word[GPE] = FullyConnectedLearner('encode', input_dim=480, output_dim=2)
    word[PER] = FullyConnectedLearner('encode', input_dim=480, output_dim=2)
    word[ORG] = FullyConnectedLearner('encode', input_dim=480, output_dim=2)
    word[LOC] = FullyConnectedLearner('encode', input_dim=480, output_dim=2)
    word[VEH] = FullyConnectedLearner('encode', input_dim=480, output_dim=2)
    word[WEA] = FullyConnectedLearner('encode', input_dim=480, output_dim=2)

    rel_phrase_contains_word['backward'] = WordToPhraseTransformer(FAC, GPE, PER, ORG, LOC, VEH, WEA,
                                                                   mode="backward", keyword="raw")
    phrase['encode'] = LastAggregationSensor("raw", edges=[rel_phrase_contains_word['backward']], map_key="encode")

    rel_pair_phrase1['backward'] = PhraseToPair('encode', mode="backward", keyword="phrase1_encode")
    rel_pair_phrase2['backward'] = PhraseToPair('encode', mode="backward", keyword="phrase2_encode")
    rel_pair_phrase1['backward'] = PhraseToPair('raw', mode="backward", keyword="phrase1_raw")
    rel_pair_phrase2['backward'] = PhraseToPair('raw', mode="backward", keyword="phrase2_raw")

    pair['index'] = PairIndexGenerator(
        'phrase1_raw', 'phrase2_raw',
        edges=[rel_pair_phrase1['backward'], rel_pair_phrase2['backward']]
    )
    pair['between_index'] = BetweenIndexGenerator('index', 'phrase1_raw', 'phrase2_raw')
    pair['phrase_features'] = MultiplyCatSensor('index', 'phrase1_encode', 'phrase2_encode')
    pair['between_encoder'] = BetweenEncoderSensor('between_index', inside=word, key='encode')
    pair['features'] = ConcatSensor('phrase_features', 'between_encoder')
    pair[ART] = FullyConnectedLearner('features', input_dim=1440, output_dim=2)
    # phrase.relate_to(FAC)[0]['selection'] = ProbabilitySelectionEdgeSensor()
    # phrase.relate_to(GPE)[0]['selection'] = ProbabilitySelectionEdgeSensor()
    # word.relate_to(PER)[0]['selection'] = ProbabilitySelectionEdgeSensor()
    # phrase.relate_to(ORG)[0]['selection'] = ProbabilitySelectionEdgeSensor()
    # phrase.relate_to(LOC)[0]['selection'] = ProbabilitySelectionEdgeSensor()
    # phrase.relate_to(VEH)[0]['selection'] = ProbabilitySelectionEdgeSensor()
    # phrase.relate_to(WEA)[0]['selection'] = ProbabilitySelectionEdgeSensor()



    from base import ACEGraph, PytorchSolverGraph, NewGraph
    #
    ACEsolver = ACEGraph(PytorchSolverGraph(NewGraph(graph)))
    return ACEsolver


#### The main entrance of the program.
def main():
    paths = ["ACE_JSON/train/result0.json"]
    updated_graph = model_declaration()

    updated_graph.train(iterations=1, paths=paths)
    # updated_graph.load()
    # updated_graph.test(paths=paths)

####
"""
This example show a full pipeline how to work with `regr`.
"""
main()
