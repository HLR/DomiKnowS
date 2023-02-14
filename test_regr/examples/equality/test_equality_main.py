import sys
import pytest
from transformers import BertTokenizer

import torch.tensor
import torch

sys.path.append('.')
sys.path.append('../../..')


def model_declaration():
    from domiknows.sensor.pytorch.sensors import TorchSensor, ReaderSensor, ConstantSensor
    from domiknows.sensor.pytorch.relation_sensors import CandidateSensor
    from domiknows.program import LearningBasedProgram
    from domiknows.program.model.pytorch import model_helper, PoiModel

    from graph import graph, word, word1, sentence, word_equal_word1, sentence_con_word #, sentence_con_word1
    from sensors import Tokenizer, TokenizerSpan
    graph.detach()

    sentence['text'] = ConstantSensor(data='This is a sample sentence to check the phrase equality or in this case the words.')
    word[sentence_con_word, 'text'] = Tokenizer(sentence['text'], relation=sentence_con_word, tokenizer=BertTokenizer.from_pretrained('bert-base-uncased'))
    word1['label'] = ConstantSensor(data=[1, 1, 1])
    word1['span'] = ConstantSensor(data=[(0, 3), (7, 12), (20, 26)])

    word['span'] = TokenizerSpan('text', tokenizer=BertTokenizer.from_pretrained('bert-base-uncased'))

    def makeSpanPairs(*_, dst, src):
        word = dst
        word1 = src
        if word.getAttribute('span')[0] == word1.getAttribute('span')[0] and word.getAttribute('span')[1] == word1.getAttribute('span')[1]:
            return True
        else:
            return False

    word[word_equal_word1.reversed] = CandidateSensor(word['span'], word1['label'], word1['span'],  forward=makeSpanPairs, relation=word_equal_word1.reversed)
        
    program = LearningBasedProgram(graph, model_helper(PoiModel, poi=[word]))
    return program


# @pytest.mark.gurobi
def test_equality_main():
    from graph import word, word1
    
    lbp = model_declaration()

    dataset = [{'data': "dummy"}, ]  # Adding the info on the reader

    for datanode in lbp.populate(dataset=dataset):
        assert datanode != None

        for child_node in datanode.getChildDataNodes():
            if child_node.getOntologyNode() != word:
                continue
            if child_node.getInstanceID() == 1:
                assert child_node.getEqualTo(conceptName = word1.name)[0].getInstanceID() == 0
                assert torch.equal(child_node.getEqualTo()[0].getAttribute("span"), torch.tensor([0, 3]))
            if child_node.getInstanceID() == 4:
                assert child_node.getEqualTo()[0].getInstanceID() == 1
                assert child_node.getEqualTo(equalName = "equalTo")[0].getInstanceID() == 1
                assert torch.equal(child_node.getEqualTo()[0].getAttribute("span"), torch.tensor([7, 12]))
                assert child_node.getAttribute("span") == (7,12)


# test_equality_main()
if __name__ == '__main__':
    pytest.main([__file__])
