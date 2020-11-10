import logging

import torch

from regr.graph import Graph, Concept, Relation
from regr.graph import ifL, notL, andL, orL, nandL
from regr.program import POIProgram
from regr.sensor.pytorch.learners import ModuleLearner
from regr.sensor.pytorch.sensors import ReaderSensor, TorchEdgeSensor, JointSensor, FunctionalSensor, FunctionalReaderSensor
from regr.sensor.pytorch.relation_sensors import EdgeSensor
from regr.program.metric import MacroAverageTracker, PRF1Tracker
from regr.program.loss import NBCrossEntropyLoss

from models import tokenize, WordEmbedding, Classifier, make_pair, concat, pair_label


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
    O = word(name='O')
    pair = Concept(name='pair')
    pair.has_a(arg1=word, arg2=word)
    work_for = pair(name='work_for')

    nandL(people, organization)
    ifL(work_for, ('x', 'y'), andL(people, 'x', organization, 'y'))

SAMPLE = {
    'text': ['John works for IBM'],
    'peop': [1, 0, 0, 0],
    'org': [0, 0, 0, 1],
    'wf': [(0,3)]
}

#
# Reader as a list
#
reader = [SAMPLE]

#
# Reader as minimal required interface
#
class Reader():
    def __iter__(self):
        yield # some sophisticated code to retrieve a sample

    def __len__(self):  # optional
        return 0 # some magic number

# reader = Reader()

#
# Sesnor / Learner
#
sentence['text'] = ReaderSensor(keyword='text')

scw = sentence.relate_to(word)[0]
word[scw.forward, 'text'] = JointSensor(sentence['text'], forward=tokenize)
word['emb'] = ModuleLearner('text', module=WordEmbedding())

word[people] = ReaderSensor(keyword='peop', label=True)
word[organization] = ReaderSensor(keyword='org', label=True)

word[people] = ModuleLearner('emb', module=Classifier())
word[organization] = ModuleLearner('emb', module=Classifier())

arg1, arg2 = pair.relate_to(word)
pair[arg1.backward, arg2.backward] = JointSensor(word['text'], forward=make_pair)
pair['emb'] = FunctionalSensor(arg1.backward('emb'), arg2.backward('emb'), forward=concat)
pair[work_for] = ModuleLearner('emb', module=Classifier(200))

pair[work_for] = FunctionalReaderSensor(pair[arg1.backward], pair[arg2.backward], keyword='wf', forward=pair_label, label=True)

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
linearsoftmax = torch.nn.Sequential(
    torch.nn.Linear(100,4),
    torch.nn.Softmax()
)
#
# datanode
#
for node in program.populate(reader, device=device):
    node.inferILPConstrains(fun=lambda val: torch.tensor(val).softmax(dim=-1).detach().cpu().numpy().tolist(), epsilon=None)
    for word_node in node.getChildDataNodes():
        print(word_node.getAttribute('index'))
        print(' - people:', word_node.getAttribute(people), 'ILP:', word_node.getAttribute(people, 'ILP'))
        print(' - organization:', word_node.getAttribute(organization), 'ILP:', word_node.getAttribute(organization, 'ILP'))
