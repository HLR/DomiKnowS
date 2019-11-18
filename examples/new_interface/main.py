'''
# Example: Entity-Mention-Relation (EMR)
## Pipeline
This example follows the pipeline we discussed in our preliminary paper.
1. Ontology Declaration
2. Model Declaration
3. Explicit inference
'''


from Graphs.Sensors.sentenceSensors import SentenceBertEmbedderSensor, \
    SentenceFlairEmbedderSensor, SentenceGloveEmbedderSensor
from Graphs.Sensors.wordSensors import WordEmbedding
from Graphs.Sensors.edgeSensors import FlairSentenceToWord, BILTransformer
from data.reader import DataLoader, ACEReader
from regr.sensor.pytorch.sensors import TorchSensor, ReaderSensor, NominalSensor, ConcatAggregationSensor, Sensor
from regr.sensor.pytorch.learners import LSTMLearner, FullyConnectedLearner

def dataloader(data_path, splitter_path):
    loader = DataLoader(data_path, splitter_path)
    loader.fire()
    return loader


def reader_declaration(loader):
    reader = ACEReader(loader.data)
    return reader


def reader_start(reader, mode):
    if mode == "train":
        return reader.readTrain()
    elif mode == "valid":
        return reader.readValid()
    elif mode == "test":
        return reader.readTest


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
    # sentence['raw'] = Sensor()
    sentence['bert'] = SentenceBertEmbedderSensor('raw')
    sentence['glove'] = SentenceGloveEmbedderSensor('raw')
    sentence['flair'] = SentenceFlairEmbedderSensor('raw')
    sentence['raw_ready'] = TorchSensor('bert', 'glove', 'flair', output='raw')

    rel_sentence_contains_word['forward'] = FlairSentenceToWord('raw_ready', mode="forward")

    word['embedding'] = WordEmbedding('raw_ready', edge=rel_sentence_contains_word['forward'])
    word['encode'] = LSTMLearner('embedding', input_dim=5220, hidden_dim=240, num_layers=1, bidirectional=True)
    word['boundary'] = FullyConnectedLearner('encode', input_dim=480, output_dim=4)
    context = {}
    print(list(word['boundary'].find(TorchSensor))[0][1](context=context))
    # word['boundary'] = ReaderSensor(keyword='boundary')

    rel_phrase_contains_word['backward'] = BILTransformer('raw_ready', 'boundary', mode="backward")

    phrase['encode'] = ConcatAggregationSensor("raw_ready", edge=rel_phrase_contains_word['backward'], map_key="encode")
    print(list(phrase['encode'].find(TorchSensor))[0][1](context=context))

    phrase[FAC] = FullyConnectedLearner('encode', input_dim=480, output_dim=2)
    phrase[GPE] = FullyConnectedLearner('encode', input_dim=480, output_dim=2)
    phrase[PER] = FullyConnectedLearner('encode', input_dim=480, output_dim=2)
    phrase[ORG] = FullyConnectedLearner('encode', input_dim=480, output_dim=2)
    phrase[LOC] = FullyConnectedLearner('encode', input_dim=480, output_dim=2)
    phrase[VEH] = FullyConnectedLearner('encode', input_dim=480, output_dim=2)
    phrase[WEA] = FullyConnectedLearner('encode', input_dim=480, output_dim=2)

    phrase[FAC] = ReaderSensor(keyword=FAC.name)
    phrase[GPE] = ReaderSensor(keyword=GPE.name)
    phrase[PER] = ReaderSensor(keyword=PER.name)
    phrase[ORG] = ReaderSensor(keyword=ORG.name)
    phrase[LOC] = ReaderSensor(keyword=LOC.name)
    phrase[VEH] = ReaderSensor(keyword=VEH.name)
    phrase[WEA] = ReaderSensor(keyword=WEA.name)


    # from Graphs.base import ACEGraph
    # # from .Graphs.base import ACEGraph
    #
    # ACEsolver = ACEGraph(PytorchSolverGraph(NewGraph(graph)))
    # ACEsolver.set_reader_instance(reader=reader)
    # return ACEsolver


#### The main entrance of the program.
def main():
    # data_path = ["LDC2006T06/ace_2005_td_v7/data/English/bc/fp1/",
    #              "LDC2006T06/ace_2005_td_v7/data/English/bc/fp2/", ]
    #              # "/home/hfaghihi/LDC2006T06/ace_2005_td_v7/data/English/bn/fp1/",
    #              # "/home/hfaghihi/LDC2006T06/ace_2005_td_v7/data/English/bn/fp2/",
    #              # "/home/hfaghihi/LDC2006T06/ace_2005_td_v7/data/English/cts/fp1/",
    #              # "/home/hfaghihi/LDC2006T06/ace_2005_td_v7/data/English/cts/fp2/",
    #              # "/home/hfaghihi/LDC2006T06/ace_2005_td_v7/data/English/nw/fp1/",
    #              # "/home/hfaghihi/LDC2006T06/ace_2005_td_v7/data/English/nw/fp2/",
    #              # "/home/hfaghihi/LDC2006T06/ace_2005_td_v7/data/English/un/fp1/",
    #              # "/home/hfaghihi/LDC2006T06/ace_2005_td_v7/data/English/un/fp2/",
    #              # "/home/hfaghihi/LDC2006T06/ace_2005_td_v7/data/English/wi/fp1/",
    #              # "/home/hfaghihi/LDC2006T06/ace_2005_td_v7/data/English/wi/fp2/", ]
    #
    # hint_path = "data"
    #
    # loader = dataloader(data_path=data_path, splitter_path=hint_path)
    # reader = reader_declaration(loader=loader)
    # train = reader_start(reader=reader, mode="train")
    # train = None
    # reader = None
    from flair.data import Sentence
    reader = [
        {
            'raw': Sentence("john works for Michigan State University"),
            'boundary': ['L', 'NONE', 'NONE', 'B', 'I', 'L'],
            'FAC': [0, 0],
            'GPE': [0, 0],
            'PER': [1, 0],
            'ORG': [0, 1],
            'LOC': [0, 0],
            'VEH': [0, 0],
            'WEA': [0, 0],
        }
    ]
    model_declaration()

    # updated_graph.train(100)

####
"""
This example show a full pipeline how to work with `regr`.
"""
main()