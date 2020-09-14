import torch

from Sensors.sensors import SentenceRepSensor
from emr.data import ConllDataLoader
from regr.graph import Graph, Concept, Relation
from regr.program import POIProgram
from regr.program.loss import NBCrossEntropyLoss
from regr.program.metric import MacroAverageTracker, PRF1Tracker
from regr.sensor.pytorch.learners import ModuleLearner
from regr.sensor.pytorch.sensors import ReaderSensor
from test_conll_reader import conllReader

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('EMR') as graph:

    sentence = Concept (name = 'sentence')
    #word = Concept(name ='word')
    org = sentence(name = 'Org')


    sentence['raw'] = ReaderSensor(keyword = 'Sentence')

    sentence['emb'] = SentenceRepSensor('raw')
   # word['raw'] = ReaderSensor(keyword= 'Word')
    sentence[org] = ReaderSensor(keyword = 'OrgLabel', label = True )


    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(96, 128)
            self.l2 = torch.nn.Linear(128, 2)

        def forward(self, x):
            a1 = self.l1(x)
            a1 = torch.nn.functional.relu(a1)
            a2 = self.l2(a1)
            return a2


    sentence[org] = ModuleLearner('emb', module = Net())

#The reader will return the whole list of learning examples each of which is a dictionary
ReaderObjectsIterator = r= conllReader("/Users/parisakordjamshidi/Documents/iGitHub/RelationalGraph/examples/emr/data/EntityMentionRelation/conll04.corp_1_train.corp",'notjason')

#The program takes the graph and learning approach as input
program = POIProgram(graph, loss= MacroAverageTracker(NBCrossEntropyLoss()), metric=PRF1Tracker())

# device options are 'cpu', 'cuda', 'cuda:x', torch.device instance, 'auto', None
device = 'auto'

program.train(ReaderObjectsIterator, train_epoch_num=1, Optim=torch.optim.Adam, device=device)
print('Training result:')
print(program.model.loss)
print(program.model.metric)

print('-'*40)

program.test(ReaderObjectsIterator, device=device)
print('Testing result:')
print(program.model.loss)
print(program.model.metric)
