import sys
sys.path.append(".")
sys.path.append("../")
sys.path.append("../../")

import logging

import torch

from domiknows.graph import Graph, Concept, Relation, DataNodeBuilder
from domiknows.graph import ifL, andL, nandL
from domiknows.program import SolverPOIProgram, POIProgram
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.sensor.pytorch.sensors import ReaderSensor, TorchEdgeSensor, JointSensor, FunctionalSensor, FunctionalReaderSensor
from domiknows.sensor.pytorch.relation_sensors import EdgeSensor
from domiknows.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.utils import setProductionLogMode, setDnSkeletonMode

from models import tokenize, WordEmbedding, Classifier, make_pair, concat, pair_label

Graph.clear()
Concept.clear()
Relation.clear()

#setProductionLogMode(no_UseTimeLog=False)
setDnSkeletonMode(True)

#
# Graph Definition
#
with Graph('example', reuse_model=True) as graph:
    sentence = Concept(name='sentence')
    word = Concept(name='word')
    sentence.contains(word)
    people = word(name='people')
    organization = word(name='organization')
    O = word(name='O')
    pair = Concept(name='pair')
    (arg1, arg2) = pair.has_a(arg1=word, arg2=word)
    work_for = pair(name='work_for')


#
# Logical Constraints
#
    # word can not be both people and organization
    nandL(people, organization)
    
    # if pair is work_for then its first pair element is people and the second is organization
    ifL(work_for('x'), andL(people(path=('x', arg1)), organization(path=('x', arg2))))
    

#
# Data Sample with labels
#
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


#
# Sensors / Learner
#
sentence['text'] = ReaderSensor(keyword='text')

scw = sentence.relate_to(word)[0]
word[scw, 'text'] = JointSensor(sentence['text'], forward=tokenize)
word['emb'] = ModuleLearner('text', scw, module=WordEmbedding())

word[people] = ReaderSensor(keyword='peop', label=True)
word[organization] = ReaderSensor(keyword='org', label=True)

word[people] = ModuleLearner('emb', module=Classifier())
word[organization] = ModuleLearner('emb', module=Classifier())

arg1, arg2 = pair.relate_to(word)
pair[arg1.reversed, arg2.reversed] = JointSensor(word['text'], forward=make_pair)
pair['emb'] = FunctionalSensor(arg1.reversed('emb'), arg2.reversed('emb'), forward=concat)
pair[work_for] = ModuleLearner('emb', module=Classifier(200))

pair[work_for] = FunctionalReaderSensor(pair[arg1.reversed], pair[arg2.reversed], keyword='wf', forward=pair_label, label=True)


#
# Defined the program
#
GBI = True
if GBI:
    program = SolverPOIProgram(graph, poi=(word, ), inferTypes=['ILP', 'local/argmax', "GBI"], 
                               loss=MacroAverageTracker(NBCrossEntropyLoss()), metric={'ILP':PRF1Tracker(DatanodeCMMetric()),
                                                                                       'argmax':PRF1Tracker(DatanodeCMMetric('local/argmax')),
                                                                                       'GBI':PRF1Tracker(DatanodeCMMetric('GBI'))})
else:
    program = SolverPOIProgram(graph, poi=(word, ), inferTypes=['ILP', 'local/argmax'], 
                               loss=MacroAverageTracker(NBCrossEntropyLoss()), metric={'ILP':PRF1Tracker(DatanodeCMMetric()),
                                                                                       'argmax':PRF1Tracker(DatanodeCMMetric('local/argmax'))})

# device options are 'cpu', 'cuda', 'cuda:x', torch.device instance, 'auto', None
device = 'auto'

# configure logging
logging.basicConfig(level=logging.INFO)


#
# Train
#
program.train(reader, train_epoch_num=10, Optim=torch.optim.Adam, device=device)
print('Training result:')
print(program.model.loss)
print(program.model.metric)

print('-'*40)

print("Model gradients after Train")
for name, x in program.model.named_parameters():
    if x.grad is None:
        print(name, 'no grad')
        continue
       
    print(name, 'grad: ', torch.sum(torch.abs(x.grad)))

#
# Test
#
program.test(reader, device=device)
print('Testing result:')
print(program.model.loss)
print(program.model.metric)

print("Model gradients after Test")
for name, x in program.model.named_parameters():
    if x.grad is None:
        print(name, 'no grad')
        continue
       
    print(name, 'grad: ', torch.sum(torch.abs(x.grad)))

print('-'*40)
linearsoftmax = torch.nn.Sequential(
    torch.nn.Linear(100,4),
    torch.nn.Softmax()
)

#
# Access learned information from Datanode 
#
DataNodeBuilder.context = "interference"
for node in program.populate(reader, device=device):
    node.infer()
    node.inferLocal()
    node.inferLocal(keys = ("local" , "normalizedProb"))
    node.inferILPResults(fun=lambda val: val.clone().detach().requires_grad_(True).softmax(dim=-1).detach().cpu().numpy().tolist(), epsilon=None)

    for word_node in node.getChildDataNodes():
        print(word_node.getAttribute('text'))
        print(' - people:', word_node.getAttribute(people), 'ILP:', word_node.getAttribute(people, 'ILP'))
        print(' - organization:', word_node.getAttribute(organization), 'ILP:', word_node.getAttribute(organization, 'ILP'))
    
    print("\nSofMax results for people - %s"%(node.collectInferredResults(people, "softmax")))
    print("ArgMax results for people - %s"%(node.collectInferredResults(people, "argmax")))

    print("\nSofMax results for organization - %s"%(node.collectInferredResults(organization, "softmax")))
    print("ArgMax results for organization - %s"%(node.collectInferredResults(organization, "argmax")))
       
    print("\nLocal/SofMax results for people - %s"%(node.collectInferredResults(people, "local/softmax").cpu()))
    print("Local/ArgMax results for people   - %s"%(node.collectInferredResults(people, "local/argmax").cpu()))
    print("ILP results for people            - %s"%(node.collectInferredResults(people, "ILP"))) 
    
    print("\nLocal/SofMax results for organization - %s"%(node.collectInferredResults(organization, "local/softmax").cpu()))
    print("Local/ArgMax results for organization   - %s"%(node.collectInferredResults(organization, "local/argmax").cpu()))
    print("ILP results for organization            - %s"%(node.collectInferredResults(organization, "ILP")))
       
#
# Verify if the learned results obey the logical constraints
#
  
    print("\nVerify Learned Results:")
    verifyResult = node.verifyResultsLC()
    if verifyResult:
        for lc in verifyResult:
            if "ifSatisfied" in verifyResult[lc]:
                if verifyResult[lc]['ifSatisfied'] == verifyResult[lc]['ifSatisfied']:
                    print("%s(if) is %i%% satisfied by learned results"%(lc, verifyResult[lc]['ifSatisfied']))
                else:
                    print("%s(if) is Nan satisfied by learned results"%(lc))
            else:
                print("%s is %i%% satisfied by learned results"%(lc, verifyResult[lc]['satisfied']))

            
    print("\nVerify ILP Results:")
    verifyResultILP = node.verifyResultsLC(key = "/ILP")
    if verifyResultILP:
        for lc in verifyResultILP:
            if "ifSatisfied" in verifyResultILP[lc]:
                if verifyResultILP[lc]['ifSatisfied'] == verifyResultILP[lc]['ifSatisfied']:
                    print("%s(if) is %i%% satisfied by ilp results"%(lc, verifyResultILP[lc]['ifSatisfied']))
                else:
                    print("%s(if) is Nan satisfied by ilp results"%(lc))
            else:
                print("%s is %i%% satisfied by ilp results"%(lc, verifyResultILP[lc]['satisfied']))