import sys
from typing import Any
import pytest

import torch

sys.path.append('.')
sys.path.append('../../..')
sys.path.append('../../../examples/ACE05')

def test_edge_main():
    from domiknows.sensor.pytorch.relation_sensors import EdgeSensor
    from domiknows.sensor.pytorch.sensors import ConstantSensor
    from graph import sentence, word, sentence_con_word, word1, word_equal_word1, graph, pair_word1, pair_word2, pair
    from domiknows.sensor.sensor import Sensor
    from domiknows.graph import DataNodeBuilder


    import spacy
    class SpacyGloveRep(EdgeSensor):
        def __init__(self, *pres, relation, edges=None, label=False, device='auto', spacy=None):
            super().__init__(*pres, relation=relation, edges=edges, label=label, device=device)
            if not spacy:
                raise ValueError('You should select a default Tokenizer')
            self.spacy = spacy

        def forward(self, text) -> Any:
            text = self.spacy(text)
            return torch.tensor([token.vector for token in text]).to(device=self.device)
#             return [token.vector for token in text]
        
        
    sensor1 = ConstantSensor(data='This is a sample sentence to check the phrase equality or in this case the words.')
    sentence['index'] = sensor1
    word['spacy'] = SpacyGloveRep('index', relation=sentence_con_word, spacy=spacy.load('en_core_web_lg'))

    data_item = DataNodeBuilder({"graph": graph})
    assert sensor1(data_item) == 'This is a sample sentence to check the phrase equality or in this case the words.'
    for sensor in word['spacy'].find(Sensor):
        sensor(data_item)
    assert len(data_item['dataNode']) == 1

if __name__ == '__main__':
    pytest.main([__file__])
