import sys
sys.path.append("../..")
import torch
from data.reader import EmailSpamReader
import os



def model_declaration():
    from domiknows.sensor.pytorch.sensors import ReaderSensor, ConcatSensor, FunctionalSensor
    from domiknows.sensor.pytorch.learners import ModuleLearner
    from domiknows.program import LearningBasedProgram
    from domiknows.program.model.pytorch import PoiModel
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
    def concat(*x): 
        return torch.cat(x, dim=-1)
    email['features'] = FunctionalSensor('subject_rep', 'body_rep', 'forward_presence', forward=concat)
    email[Spam] = ModuleLearner('features', module=nn.Linear(601, 2))
    email[Regular] = ModuleLearner('features', module=nn.Linear(601, 2))
    email[Spam] = ReaderSensor(keyword='Spam', label=True)
    email[Regular] = ReaderSensor(keyword='Regular', label=True)

    from domiknows.program import POIProgram, IMLProgram, SolverPOIProgram
    from domiknows.program.metric import MacroAverageTracker, PRF1Tracker, PRF1Tracker, DatanodeCMMetric
    from domiknows.program.loss import NBCrossEntropyLoss

    program = SolverPOIProgram(graph, inferTypes=['ILP', 'local/argmax'], loss=MacroAverageTracker(NBCrossEntropyLoss()), metric={'ILP':PRF1Tracker(DatanodeCMMetric()),'argmax':PRF1Tracker(DatanodeCMMetric('local/argmax'))})

    return program


def main():
    from graph import email, Spam, Regular

    # set logger level to see training and testing logs
    import logging
    logging.basicConfig(level=logging.INFO)

    lbp = model_declaration()

    pwd = os.getcwd()
    
    train_dataset = EmailSpamReader(file='data/train', type="folder")  # Adding the info on the reader
    test_dataset = EmailSpamReader(file='data/test', type="folder")

    lbp.train(train_dataset, test_set=test_dataset, train_epoch_num=5, Optim=torch.optim.Adam, device='auto')
    lbp.test(test_dataset, device="auto")

    for datanode in lbp.populate(test_dataset):
        print('datanode:', datanode)
        print('Spam:', datanode.getAttribute(Spam))
        print('Regular:', datanode.getAttribute(Regular))
        print('inference spam:', datanode.getAttribute(Spam, 'ILP'))
        print('inference regular:', datanode.getAttribute(Regular, 'ILP'))


if __name__ == '__main__':
    main()

