import sys
import pytest
from transformers import BertTokenizer

sys.path.append('.')
sys.path.append('../../..')


def model_declaration():
    from regr.sensor.pytorch.sensors import TorchSensor, ReaderSensor, TorchEdgeReaderSensor, ForwardEdgeSensor, \
        ConstantSensor
    from regr.sensor.pytorch.query_sensor import CandidateEqualSensor
    from regr.program import LearningBasedProgram
    from regr.program.model.pytorch import model_helper, PoiModel

    from graph import graph, word, word1, sentence, word_equal_word1, sentence_con_word
    from sensors import Tokenizer, TokenizerSpan
    graph.detach()

    # --- City
    sentence['index'] = ConstantSensor(data='This is a sample sentence to check the phrase equality or in this case the words.')
    sentence_con_word['forward'] = Tokenizer('index', to='index', mode='forward', tokenizer=BertTokenizer.from_pretrained('bert-base-uncased'))
    word1['index'] = ConstantSensor(data=['phrase1', 'phrase2', 'phrase3'])
    word1['span'] = ConstantSensor(data=[(0, 0), (2, 4), (9, 10)])
    word['span'] = TokenizerSpan('index', edges=[sentence_con_word['forward']], tokenizer=BertTokenizer.from_pretrained('bert-base-uncased'))

    def makeSpanPairs(current_spans, phrase1, phrase2):
        if phrase1.getAttribute('span') == phrase2.getAttribute('span'):
            return True
        else:
            return False

    word['match'] = CandidateEqualSensor(forward=makeSpanPairs)

    program = LearningBasedProgram(graph, model_helper(PoiModel, poi=[word['match']]))
    return program


@pytest.mark.gurobi
def test_graph_coloring_main():
    lbp = model_declaration()

    dataset = [{'data': "dummy"}, ]  # Adding the info on the reader

    for datanode in lbp.populate(dataset=dataset):
        assert datanode != None

        for child_node in datanode.getChildDataNodes():
            print(child_node)


if __name__ == '__main__':
    pytest.main([__file__])
