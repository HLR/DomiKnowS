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
    SentenceFlairEmbedderSensor, SentenceGloveEmbedderSensor, FlairSentenceSensor
from Graphs.Sensors.wordSensors import WordEmbedding
from Graphs.Sensors.edgeSensors import FlairSentenceToWord, BILTransformer
from regr.sensor.pytorch.sensors import TorchSensor, ReaderSensor, NominalSensor, ConcatAggregationSensor, ProbabilitySelectionEdgeSensor, MaxAggregationSensor
from regr.sensor.pytorch.learners import LSTMLearner, FullyConnectedLearner
from data.reader import SimpleReader


def model_declaration():
    from Graphs.graph import graph, rel_phrase_contains_word, rel_sentence_contains_phrase, rel_sentence_contains_word

    print("model started")
    graph.detach()

    sentence = graph['linguistic/sentence']
    word = graph['linguistic/word']
    phrase = graph['linguistic/phrase']

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
    sentence['raw_ready'] = TorchSensor('bert', 'glove', 'flair', output='flair_sentence')

    rel_sentence_contains_word['forward'] = FlairSentenceToWord('raw_ready', mode="forward")

    word['embedding'] = WordEmbedding('raw_ready', edge=rel_sentence_contains_word['forward'])
    word['encode'] = LSTMLearner('embedding', input_dim=5220, hidden_dim=240, num_layers=1, bidirectional=True)
    # word['boundary'] = FullyConnectedLearner('encode', input_dim=480, output_dim=4)
    # word['boundary'] = ReaderSensor(keyword='boundary')

    # rel_phrase_contains_word['backward'] = BILTransformer('raw_ready', 'boundary', mode="backward")

    # phrase['encode'] = MaxAggregationSensor("raw_ready", edge=rel_phrase_contains_word['backward'], map_key="encode")

    word[FAC] = FullyConnectedLearner('encode', input_dim=480, output_dim=2)
    word[GPE] = FullyConnectedLearner('encode', input_dim=480, output_dim=2)
    word[PER] = FullyConnectedLearner('encode', input_dim=480, output_dim=2)
    word[ORG] = FullyConnectedLearner('encode', input_dim=480, output_dim=2)
    word[LOC] = FullyConnectedLearner('encode', input_dim=480, output_dim=2)
    word[VEH] = FullyConnectedLearner('encode', input_dim=480, output_dim=2)
    word[WEA] = FullyConnectedLearner('encode', input_dim=480, output_dim=2)

    word[FAC] = ReaderSensor(keyword=FAC.name, label=True)
    word[GPE] = ReaderSensor(keyword=GPE.name, label=True)
    word[PER] = ReaderSensor(keyword=PER.name, label=True)
    word[ORG] = ReaderSensor(keyword=ORG.name, label=True)
    word[LOC] = ReaderSensor(keyword=LOC.name, label=True)
    word[VEH] = ReaderSensor(keyword=VEH.name, label=True)
    word[WEA] = ReaderSensor(keyword=WEA.name, label=True)

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
    paths = ["ACE_JSON/train/result0.json", "ACE_JSON/train/result1.json"]
    updated_graph = model_declaration()
    updated_graph.train(iterations=2, paths=paths)
    # updated_graph.load()
    # updated_graph.test(reader=reader)

####
"""
This example show a full pipeline how to work with `regr`.
"""
main()
