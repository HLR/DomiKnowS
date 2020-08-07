import sys
import pytest

sys.path.append('.')
sys.path.append('../..')


def model_declaration():
    from regr.sensor.pytorch.sensors import ReaderSensor, TorchEdgeReaderSensor, ForwardEdgeSensor, ConstantSensor, ConcatSensor
    from regr.sensor.pytorch.learners import ModuleLearner
    from regr.sensor.pytorch.query_sensor import CandidateReaderSensor
    from regr.program import LearningBasedProgram
    from regr.program.model.pytorch import PoiModel
    import torch
    from torch import nn

    from graph import graph, email, Spam, Regular
    from Sensors.sensors import SentenceRepSensor, ForwardPresenceSensor

    graph.detach()

    # --- City
    email['subject'] = ReaderSensor(keyword='Subject')
    email['body'] = ReaderSensor(keyword="Body")
    email['forward_subject'] = ReaderSensor(keyword="ForwardSubject")
    email['forward_body'] = ReaderSensor(keyword="ForwardBody")
    email['subject_rep'] = SentenceRepSensor('subject')
    email['body_rep'] = SentenceRepSensor('body')
    email['forward_presence'] = ForwardPresenceSensor('forward_body')
    email['features'] = ConcatSensor('subject_rep', 'body_rep', 'forward_presence')
    email[Spam] = ModuleLearner('features', module=nn.Linear(601, 2))
    email[Regular] = ModuleLearner('features', module=nn.Linear(601, 2))
    email[Spam] = ReaderSensor(keyword='Spam', label=True)
    email[Regular] = ReaderSensor(keyword='Regular', label=True)

    program = LearningBasedProgram(graph, PoiModel)
    return program


@pytest.mark.gurobi
def test_graph_coloring_main():
    from data.reader import EmailSpamReader
    from graph import email, Spam, Regular

    lbp = model_declaration()

    dataset = EmailSpamReader(file='data/train', type="folder").run()  # Adding the info on the reader

    for datanode in lbp.populate(dataset=dataset, inference=True):
        assert datanode != None
        # assert len(datanode.getChildDataNodes()) == 9

        # call solver
        conceptsRelations = (Spam, Regular)
        datanode.inferILPConstrains(*conceptsRelations, fun=None, minimizeObjective=False)

        s = datanode.getAttribute(Spam, 'ILP').item()
        f = datanode.getAttribute(Regular, 'ILP').item()
        if f > 0:
            assert s == 0
        else:
            assert s == 1

if __name__ == '__main__':
    pytest.main([__file__])
