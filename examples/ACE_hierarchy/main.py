'''
# Example: Entity-Mention-Relation (EMR)
## Pipeline
This example follows the pipeline we discussed in our preliminary paper.
1. Ontology Declaration
2. Model Declaration
3. Explicit inference
'''
import sys
sys.path.append("../hierarchyACE")
sys.path.append("../..")

from Graphs.Sensors.sentenceSensors import SentenceBertEmbedderSensor, \
    SentenceFlairEmbedderSensor, SentenceGloveEmbedderSensor, FlairSentenceSensor, SentencePosTagger
from Graphs.Sensors.wordSensors import WordEmbedding, WordPosTaggerSensor
from Graphs.Sensors.edgeSensors import FlairSentenceToWord, SentenceToWordPos
from regr.sensor.pytorch.sensors import TorchSensor, ReaderSensor, ListConcator
from regr.sensor.pytorch.learners import LSTMLearner, FullyConnectedLearner, FullyConnectedLearnerRelu


def model_declaration():
    from Graphs.graph import graph, rel_sentence_contains_word
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

    # FAC SUB
    Airport = graph["application/Airport"]
    Building_Grounds = graph["application/Building-Grounds"]
    Path = graph["application/Path"]
    Plant = graph["application/Plant"]
    Subarea_Facility = graph["application/Subarea-Facility"]

    # GPE SUB
    Continent = graph["application/Continent"]
    County_or_District = graph["application/County-or-District"]
    GPE_Cluster = graph["application/GPE-Cluster"]
    Nation = graph["application/Nation"]
    Population_Center = graph["application/Population-Center"]
    Special = graph["application/Special"]
    State_or_Province = graph["application/State-or-Province"]

    # LOC SUB
    Address = graph["application/Address"]
    Boundary = graph["application/Boundary"]
    Celestial = graph["application/Celestial"]
    Land_Region_Natural = graph["application/Land-Region-Natural"]
    Region_General = graph["application/Region-General"]
    Region_International = graph["application/Region-International"]
    Water_Body = graph["application/Water-Body"]

    # ORG SUB
    Commercial = graph["application/Commercial"]
    Educational = graph["application/Educational"]
    Entertainment = graph["application/Entertainment"]
    Government = graph["application/Government"]
    Media = graph["application/Media"]
    Medical_Science = graph["application/Medical-Science"]
    Non_Governmental = graph["application/Non-Governmental"]
    Religious = graph["application/Religious"]
    Sports = graph["application/Sports"]

    # PER SUB
    Group = graph["application/Group"]
    Indeterminate = graph["application/Indeterminate"]
    Individual = graph["application/Individual"]

    # VEH SUB
    Air = graph["application/Air"]
    Land = graph["application/Land"]
    Subarea_Vehicle = graph["application/Subarea-Vehicle"]
    Underspecified = graph["application/Underspecified"]
    Water = graph["application/Water"]

    # WEA SUB
    Biological = graph["application/Biological"]
    Blunt = graph["application/Blunt"]
    Chemical = graph["application/Chemical"]
    Exploding = graph["application/Exploding"]
    Nuclear = graph["application/Nuclear"]
    Projectile = graph["application/Projectile"]
    Sharp = graph["application/Sharp"]
    Shooting = graph["application/Shooting"]
    WEA_Underspecified = graph["application/WEA-Underspecified"]

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
    word['encode'] = LSTMLearner('features', input_dim=5236, hidden_dim=500, num_layers=1, bidirectional=True)

    word['encode_final'] = FullyConnectedLearnerRelu('encode', input_dim=1000, output_dim=1000)

    word[FAC] = ReaderSensor(keyword=FAC.name, label=True)
    word[GPE] = ReaderSensor(keyword=GPE.name, label=True)
    word[PER] = ReaderSensor(keyword=PER.name, label=True)
    word[ORG] = ReaderSensor(keyword=ORG.name, label=True)
    word[LOC] = ReaderSensor(keyword=LOC.name, label=True)
    word[VEH] = ReaderSensor(keyword=VEH.name, label=True)
    word[WEA] = ReaderSensor(keyword=WEA.name, label=True)

    word[FAC] = FullyConnectedLearner('encode_final', input_dim=1000, output_dim=2)
    word[GPE] = FullyConnectedLearner('encode_final', input_dim=1000, output_dim=2)
    word[PER] = FullyConnectedLearner('encode_final', input_dim=1000, output_dim=2)
    word[ORG] = FullyConnectedLearner('encode_final', input_dim=1000, output_dim=2)
    word[LOC] = FullyConnectedLearner('encode_final', input_dim=1000, output_dim=2)
    word[VEH] = FullyConnectedLearner('encode_final', input_dim=1000, output_dim=2)
    word[WEA] = FullyConnectedLearner('encode_final', input_dim=1000, output_dim=2)

    _subnames = [Airport, Building_Grounds, Path, Subarea_Facility, GPE_Cluster,
                 Nation, Population_Center, State_or_Province, Boundary, Celestial,
                 Land_Region_Natural, Region_General, Region_International, Water_Body, Commercial, Educational,
                 Government, Media, Non_Governmental, Group,
                 Indeterminate, Individual, Land, Water, Chemical,
                 Exploding, Nuclear, Projectile, Shooting, WEA_Underspecified]

    for item in _subnames:
        word[item] = ReaderSensor(keyword=item.name, label=True)
        word[item] = FullyConnectedLearner('encode_final', input_dim=1000, output_dim=2)

    from base import ACEGraph, PytorchSolverGraph, NewGraph
    #
    ACEsolver = ACEGraph(PytorchSolverGraph(NewGraph(graph)))
    return ACEsolver


#### The main entrance of the program.
def main():
    # paths = ["ACE_JSON/test/result0.json", "ACE_JSON/test/result1.json", "ACE_JSON/test/result2.json"]
    # paths = ["ACE_JSON/train/result0.json", "ACE_JSON/train/result1.json", "ACE_JSON/train/result2.json", "ACE_JSON/train/result3.json", "ACE_JSON/train/result4.json", "ACE_JSON/train/result5.json", "ACE_JSON/train/result6.json", "ACE_JSON/train/result7.json", "ACE_JSON/train/result8.json", "ACE_JSON/train/result9.json", "ACE_JSON/train/result10.json"]
    paths = ["ACE_JSON/train/result0.json"]
    # paths = ["ACE_JSON/train/result0.json"]
    updated_graph = model_declaration()
    #
    # # updated_graph.load()
    updated_graph.structured_train_constraint(iterations=50, paths=paths, ratio=1)
    # updated_graph.load()
    paths = ["ACE_JSON/test/result0.json", "ACE_JSON/test/result1.json", "ACE_JSON/test/result2.json"]
    updated_graph.predConstraint(paths=paths)

####
"""
This example show a full pipeline how to work with `regr`.
"""
main()
