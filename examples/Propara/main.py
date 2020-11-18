import sys
import torch
# from data.reader import EmailSpamReader

sys.path.append('.')
sys.path.append('../..')


def model_declaration():
    from regr.sensor.pytorch.sensors import ReaderSensor, TorchEdgeReaderSensor, ForwardEdgeSensor, ConstantSensor
    from regr.sensor.pytorch.learners import ModuleLearner
    from regr.program import LearningBasedProgram
    from regr.program.model.pytorch import PoiModel
    import torch
    from torch import nn

    from graph import graph, procedure, word, step, entity, entity_step, entity_step_word, location_start, location_end, non_existence, unknown_loc, known_loc, action, create, destroy, other
    from graph import entity_of_step, entity_of_step_word, step_contains_word, step_of_entity, step_of_entity_word, word_of_entity_step, procedure_contain_step, action_arg1, action_arg2

    graph.detach()

    # --- City
    procedure['id'] = ReaderSensor(keyword='procedureID')
    step[procedure_contain_step, 'number', 'text'] = ReaderSensor(keyword='steps')
    # word[step_contains_word, 'raw'] = ReaderSensor(keyword='words')
    entity['raw'] = ReaderSensor(keyword='entity')
    entity_step[entity_of_step, step_of_entity] = ReaderSensor(keyword='entity_step')
    entity_step[non_existence] = ReaderSensor(keyword='non_existence')
    entity_step[unknown_loc] = ReaderSensor(keyword='unknown_location')
    entity_step[known_loc] = ReaderSensor(keyword='known_location')
    action[action_arg1, action_arg2] = ReaderSensor(keyword='action')
    action[create] = ReaderSensor(keyword='create')
    action[destroy] = ReaderSensor(keyword='destroy')
    action[other] = ReaderSensor(keyword='other')
    # entity_step_word[entity_of_step_word, step_of_entity_word, word_of_entity_step] = ReaderSensor(keyword='entity_step_word')
    # entity_step_word[location_start] = ReaderSensor(keyword='location_start')
    # entity_step_word[location_end] = ReaderSensor(keyword='location_end')

    program = LearningBasedProgram(graph, PoiModel)
    return program


def main():
    from graph import graph, procedure, word, step, entity, entity_step, entity_step_word, location_start, location_end, non_existence, unknown_loc, known_loc

    # set logger level to see training and testing logs
    import logging
    logging.basicConfig(level=logging.INFO)

    lbp = model_declaration()

    dataset = EmailSpamReader(file='data/train', type="folder")  # Adding the info on the reader

    lbp.train(dataset, train_epoch_num=30, Optim=torch.optim.Adam, device='auto')

    for datanode in lbp.populate(dataset=dataset):
        print('datanode:', datanode)
        print('Spam:', datanode.getAttribute(Spam).softmax(-1))
        print('Regular:', datanode.getAttribute(Regular).softmax(-1))
        datanode.inferILPConstrains(fun=lambda val: torch.tensor(val).softmax(dim=-1).detach().cpu().numpy().tolist(),
                                    epsilon=None)
        print('inference spam:', datanode.getAttribute(Spam, 'ILP'))
        print('inference regular:', datanode.getAttribute(Regular, 'ILP'))


main()

