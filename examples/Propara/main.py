import sys
import torch
# from data.reader import EmailSpamReader

sys.path.append('.')
sys.path.append('../..')

from regr.sensor.pytorch.sensors import ReaderSensor, JointSensor
from regr.sensor.pytorch.relation_sensors import EdgeSensor
from regr.sensor.pytorch.sensors import ReaderSensor
from regr.program import LearningBasedProgram
from regr.program.model.pytorch import PoiModel
import torch
from torch import nn

class EdgeReaderSensor(EdgeSensor, ReaderSensor):
    def __init__(self, *pres, relation, mode="forward", keyword=None, **kwargs):
        super().__init__(*pres, relation=relation, mode=mode, **kwargs)
        self.keyword = keyword
        self.data = None
        
class JoinReaderSensor(JointSensor, ReaderSensor):
    pass
            
class JoinEdgeReaderSensor(JointSensor, EdgeReaderSensor):
    pass



def model_declaration():
    from graph import graph, procedure, word, step, entity, entity_step, entity_step_word, location_start, location_end, non_existence, unknown_loc, known_loc, action, create, destroy, other
    from graph import entity_of_step, entity_of_step_word, step_contains_word, step_of_entity, step_of_entity_word, word_of_entity_step, procedure_contain_step, action_arg1, action_arg2

    graph.detach()

    # --- City
    procedure['id'] = ReaderSensor(keyword='procedureID')
    step[procedure_contain_step.forward, 'text'] = JoinEdgeReaderSensor(procedure['id'], keyword='steps', relation=procedure_contain_step, mode="forward")
    # word[step_contains_word, 'raw'] = ReaderSensor(keyword='words')
    entity['raw'] = ReaderSensor(keyword='entities')
    entity_step[entity_of_step.forward, step_of_entity.forward] = JoinReaderSensor(entity['raw'], step['text'], step[procedure_contain_step.forward], keyword='entity_step')
    entity_step[non_existence] = ReaderSensor(keyword='non_existence')
    entity_step[unknown_loc] = ReaderSensor(keyword='known')
    entity_step[known_loc] = ReaderSensor(keyword='unkown')
    entity_step[non_existence] = ReaderSensor(keyword='non_existence')
    entity_step[unknown_loc] = ReaderSensor(keyword='known')
    entity_step[known_loc] = ReaderSensor(keyword='unkown')
    action[action_arg1.forward, action_arg2.forward] = JoinReaderSensor(entity_step[entity_of_step.forward], entity_step[step_of_entity.forward], keyword='action')
    action[create] = ReaderSensor(action_arg1.forward, action_arg2.forward, keyword='create')
    action[destroy] = ReaderSensor(action_arg1.forward, action_arg2.forward, keyword='destroy')
    action[other] = ReaderSensor(action_arg1.forward, action_arg2.forward, keyword='other')
    action[create] = ReaderSensor(action_arg1.forward, action_arg2.forward, keyword='create')
    action[destroy] = ReaderSensor(action_arg1.forward, action_arg2.forward, keyword='destroy')
    action[other] = ReaderSensor(action_arg1.forward, action_arg2.forward, keyword='other')
    # entity_step_word[entity_of_step_word, step_of_entity_word, word_of_entity_step] = ReaderSensor(keyword='entity_step_word')
    # entity_step_word[location_start] = ReaderSensor(keyword='location_start')
    # entity_step_word[location_end] = ReaderSensor(keyword='location_end')
    program = LearningBasedProgram(graph, **{
        'Model': PoiModel,
#         'poi': (known_loc, unknown_loc, non_existence, other, destroy, create),
        'loss': None,
        'metric': None,
    })
    return program


def main():
    from graph import graph, procedure, word, step, entity, entity_step, entity_step_word, location_start, location_end, non_existence, unknown_loc, known_loc, action, create, destroy, other
    from graph import entity_of_step, entity_of_step_word, step_contains_word, step_of_entity, step_of_entity_word, word_of_entity_step, procedure_contain_step, action_arg1, action_arg2

    # set logger level to see training and testing logs
    import logging
    logging.basicConfig(level=logging.INFO)

    lbp = model_declaration()

    dataset = ProparaReader(file='test_data.json')  # Adding the info on the reader

#     lbp.test(dataset, device='auto')

    for datanode in lbp.populate(dataset, device="cpu"):
        print('datanode:', datanode)
#         print('Spam:', datanode.getAttribute(Spam).softmax(-1))
#         print('Regular:', datanode.getAttribute(Regular).softmax(-1))
        datanode.inferILPResults(fun=lambda val: torch.tensor(val).softmax(dim=-1).detach().cpu().numpy().tolist(), epsilon=None)
        print('datanode:', datanode)
#         print('inference spam:', datanode.getAttribute(Spam, 'ILP'))
#         print('inference regular:', datanode.getAttribute(Regular, 'ILP'))


main()

