import sys
from typing import Any
import pytest
from transformers import BertTokenizer

import torch.tensor
import torch


sys.path.append('.')
sys.path.append('../../..')
sys.path.append('../../../examples/ACE05')


def test_edge_main():
    from sensors import TorchEdgeSensor
    from regr.sensor.pytorch.sensors import ConstantSensor, JointSensor
    from regr.sensor.pytorch.query_sensor import CandidateEqualSensor, CandidateRelationSensor
    from graph import sentence, word, sentence_con_word, word1, word_equal_word1, graph, pair_word1, pair_word2, pair
    from regr.sensor.sensor import Sensor
    from regr.graph import DataNodeBuilder


    class SampleEdge(TorchEdgeSensor):
        def forward(self, ) -> Any:
            return self.inputs[0].split(" "), [1,2,3,4]
    sensor1 = ConstantSensor(data='This is a sample sentence to check the phrase equality or in this case the words.')
    sentence['index'] = sensor1
    sentence['joint1', 'joint2'] = JointSensor(forward=lambda : ((1,2,3), ('a', 'b', 'c')))
    # word['index', 'ids'] = SampleEdge('index', relation=sentence_con_word, mode="forward")
    # word1['index'] = ConstantSensor(data=['words', 'case', 'quality', 'is'])

    data_item = DataNodeBuilder({"graph": graph})
    assert sensor1(data_item) == 'This is a sample sentence to check the phrase equality or in this case the words.'
    assert sentence['joint1'](data_item) == (1,2,3)
    assert sentence['joint2'](data_item) == ('a', 'b', 'c')
    # for sensor in word['index'].find(Sensor):
    #     sensor(data_item)
    # for sensor in word1['index'].find(Sensor):
    #     sensor(data_item)
    print(data_item)

if __name__ == '__main__':
    pytest.main([__file__])
