import sys
sys.path.append("../..")
import torch
from data.reader import EmailSpamReader
import os



def model_declaration():
    from regr.sensor.pytorch.sensors import ReaderSensor, ConcatSensor, FunctionalSensor
    from regr.sensor.pytorch.learners import ModuleLearner
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
    def concat(*x): 
        return torch.cat(x, dim=-1)
    email['features'] = FunctionalSensor('subject_rep', 'body_rep', 'forward_presence', forward=concat)
    email[Spam] = ModuleLearner('features', module=nn.Linear(193, 2))
    email[Regular] = ModuleLearner('features', module=nn.Linear(193, 2))
    email[Spam] = ReaderSensor(keyword='Spam', label=True)
    email[Regular] = ReaderSensor(keyword='Regular', label=True)

    program = LearningBasedProgram(graph, PoiModel)
    return program


def test_main():
    from graph import email, Spam, Regular

    # set logger level to see training and testing logs
    import logging
    logging.basicConfig(level=logging.INFO)

    lbp = model_declaration()

    pwd = os.getcwd()
    
    dataset = EmailSpamReader(file='examples/Email_Spam/data/train', type="folder")  # Adding the info on the reader

    lbp.train(dataset, train_epoch_num=5, Optim=torch.optim.Adam, device='auto')

    for datanode in lbp.populate(dataset=dataset):
        print('datanode:', datanode)
        print('Spam:', datanode.getAttribute(Spam))
        print('Regular:', datanode.getAttribute(Regular))
        datanode.inferILPResults(Spam, Regular, epsilon=None)
        print('inference spam:', datanode.getAttribute(Spam, 'ILP'))
        print('inference regular:', datanode.getAttribute(Regular, 'ILP'))


test_main()

