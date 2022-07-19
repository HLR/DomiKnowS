import sys

from regr.program.lossprogram import SampleLossProgram

sys.path.append('.')
sys.path.append('../..')

import torch,argparse
from transformers import AdamW
from regr.program.loss import NBCrossEntropyLoss, BCEWithLogitsIMLoss
from regr.program.metric import MacroAverageTracker, PRF1Tracker, MetricTracker, CMWithLogitsMetric, DatanodeCMMetric
import logging
from reader import read_data
from regr.graph import Graph, Concept, Relation, ifL, andL
from regr.program.primaldualprogram import PrimalDualProgram
from regr.sensor.pytorch import ModuleLearner
from regr.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from regr.sensor.pytorch.sensors import ReaderSensor, JointSensor, FunctionalSensor
from utils import Generator, make_facts, label_reader, RobertaTokenizer, BBRobert
from regr.program import SolverPOIProgram, IMLProgram
from regr.program.model.pytorch import SolverModel, IMLModel

parser = argparse.ArgumentParser(description='Run beleifebank Main Learning Code')

parser.add_argument('--cuda', dest='cuda_number', default=0, help='cuda number to train the models on',type=int)
parser.add_argument('--epoch', dest='cur_epoch', default=2, help='number of epochs you want your model to train on',type=int)

parser.add_argument('--samplenum', dest='samplenum', default=10, help='sample sizes for low data regime 10,20,40 max 37',type=int)

parser.add_argument('--pd', dest='primaldual', default=False, help='whether or not to use primaldual constriant learning',type=bool)
parser.add_argument('--iml', dest='IML', default=False, help='whether or not to use IML constriant learning',type=bool)
parser.add_argument('--sam', dest='SAM', default=False, help='whether or not to use sampling learning',type=bool)

parser.add_argument('--batch', dest='batch_size', default=32, help='batch size for neural network training',type=int)
parser.add_argument('--beta', dest='beta', default=0.5, help='primal dual or IML multiplier',type=float)
parser.add_argument('--lr', dest='learning_rate', default=2e-6, help='learning rate of the adam optimiser',type=float)

args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

calibration_data,silver_data,constraints=read_data(batch_size=32*1,sample_size=args.samplenum)

cuda_number= args.cuda_number
device = "cuda:"+str(cuda_number) if torch.cuda.is_available() else 'cpu'

print("device is : ",device)

def guess_pair(sentence, arg1, arg2):

    if len(sentence)<2 or arg1==arg2:
        return False
    sentence1, sentence2 = arg1.getAttribute('sentence'), arg2.getAttribute('sentence')
    if sentence1 in constraints and sentence2 in constraints[sentence1]:
        return True
    else:
        return False

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('belief_bank') as graph:
    subject = Concept(name='subject')
    facts = Concept(name='facts')
    subject_facts_contains, = subject.contains(facts)

    fact_check = facts(name='fact_check')
    direction = Concept(name='direction')
    i_arg1, i_arg2 = direction.has_a(arg1=facts, arg2=facts)

    ifL(fact_check('x'), fact_check(path=('x', direction, i_arg2)))

subject['name'] = ReaderSensor(keyword='name')
subject['facts'] = ReaderSensor(keyword='facts')
subject['labels'] = ReaderSensor(keyword='labels')

facts[subject_facts_contains,"name", "sentence", 'label'] = JointSensor(\
    subject['name'], subject['facts'], subject['labels'],forward=make_facts)
facts[fact_check] = FunctionalSensor(subject_facts_contains, "label", forward=label_reader, label=True)

direction[i_arg1.reversed, i_arg2.reversed] = CompositionCandidateSensor(facts['sentence'],relations=(i_arg1.reversed, i_arg2.reversed),forward=guess_pair)

facts["token_ids", "Mask"] = JointSensor("name", "sentence", forward=RobertaTokenizer(),device=device)
facts[fact_check] = ModuleLearner("token_ids", "Mask", module=BBRobert())

if not args.primaldual and not args.IML and not args.SAM:
    program = SolverPOIProgram(graph, poi=[facts[fact_check],direction],inferTypes=['ILP','local/argmax'],\
                    loss=MacroAverageTracker(NBCrossEntropyLoss()),metric={'ILP': PRF1Tracker(DatanodeCMMetric()),\
                                                'softmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})
elif args.primaldual:
    program = PrimalDualProgram(graph,SolverModel, poi=[facts[fact_check],direction],inferTypes=['ILP','local/argmax'],\
                    loss=MacroAverageTracker(NBCrossEntropyLoss()),metric={'ILP': PRF1Tracker(DatanodeCMMetric()),\
                                               'softmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},beta=0.3)
elif args.IML:
    program = IMLProgram(graph, poi=[facts[fact_check],direction],inferTypes=['ILP','local/argmax'],\
                   loss=MacroAverageTracker(BCEWithLogitsIMLoss(lmbd=0.5)),metric={'ILP': PRF1Tracker(DatanodeCMMetric()),\
                                               'softmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})
elif args.SAM:
    program = SampleLossProgram(graph, SolverModel,poi=[facts[fact_check],direction],inferTypes=['ILP','local/argmax'],
        metric={'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},loss=MacroAverageTracker(NBCrossEntropyLoss()),sample=True,sampleSize=300,sampleGlobalLoss=True)

program.train(calibration_data,valid_set=silver_data, train_epoch_num=args.cur_epoch, Optim=lambda param: AdamW(param, lr = args.learning_rate ,eps = 1e-9 ), device=device)

_,silver_data_test,constraints=read_data(batch_size=32*16,sample_size=40)

program.test(silver_data_test, device=device)