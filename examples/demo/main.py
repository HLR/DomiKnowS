import logging

import torch

from regr.graph import Graph, Concept, Relation
from regr.graph import ifL, notL, andL, orL, nandL
from regr.program import POIProgram
from regr.sensor.pytorch.learners import ModuleLearner
from regr.sensor.pytorch.sensors import ReaderSensor, TorchEdgeSensor
from regr.program.metric import MacroAverageTracker, PRF1Tracker
from regr.program.loss import NBCrossEntropyLoss

from models import tokenize, WordEmbedding, Classifier


Graph.clear()
Concept.clear()
Relation.clear()


#
# graph
#
with Graph('example') as graph:
    sentence = Concept(name='sentence')
    word = Concept(name='word')
    sentence.contains(word)
    people = word(name='people')
    organization = word(name='organization')
    nandL(people, organization)
    # pair = Concept(name='pair')
    # pair.has_a(arg1=word, arg2=word)
    # work_for = pair(name='work_for')
    # ifL(work_for, ('x', 'y'), andL(people, 'x', organization, 'y'))

#
# Reader as a list
#
reader = [{
    'text': 'John works for IBM',
    'peop': [1, 0, 0, 0],
    'org': [0, 0, 0, 1],
    'wf': [(0,3)]
    },]

#
# Reader as minimal required interface
#
SAMPLE = {
    'text': 'John works for IBM',
    'peop': [1, 0, 0, 0],
    'org': [0, 0, 0, 1],
    'wf': [(0,3)]
}
class Reader():
    def __iter__(self):
        yield SAMPLE

    def __len__(self):  # optional
        return 1

reader = Reader()

#
# Sesnor / Learner
#
sentence['index'] = ReaderSensor(keyword='text')

scw = sentence.relate_to(word)[0]
scw['forward'] = TorchEdgeSensor('index', to='index', forward=tokenize)
# word['index'] = ContainEdgeSensor('index', forward=tokenize)

word[people] = ReaderSensor(keyword='peop', label=True)
word[organization] = ReaderSensor(keyword='org', label=True)


word['emb'] = ModuleLearner('index', module=WordEmbedding())
word[people] = ModuleLearner('emb', module=Classifier())
word[organization] = ModuleLearner('emb', module=Classifier())

#
# Program
#
program = POIProgram(graph, loss=MacroAverageTracker(NBCrossEntropyLoss()), metric=PRF1Tracker())

# device options are 'cpu', 'cuda', 'cuda:x', torch.device instance, 'auto', None
device = 'auto'

logging.basicConfig(level=logging.INFO)

#
# train
#
program.train(reader, train_epoch_num=100, Optim=torch.optim.Adam, device=device)
print('Training result:')
print(program.model.loss)
print(program.model.metric)

print('-'*40)

#
# test
#
program.test(reader, device=device)
print('Testing result:')
print(program.model.loss)
print(program.model.metric)

print('-'*40)

#
# datanode
#
for node in program.populate(reader, device=device):
    node.inferILPConstrains(fun=lambda val: torch.tensor(val).softmax(dim=-1).detach().cpu().numpy().tolist(), epsilon=None)
    for word_node in node.getChildDataNodes():
        print(word_node.getAttribute('index'))
        print(' - people:', word_node.getAttribute(people), 'ILP:', word_node.getAttribute(people, 'ILP'))
        print(' - organization:', word_node.getAttribute(organization), 'ILP:', word_node.getAttribute(organization, 'ILP'))
