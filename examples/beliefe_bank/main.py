import sys

sys.path.append('.')
sys.path.append('../..')

from regr.program.lossprogram import SampleLossProgram
import torch,argparse
from transformers import AdamW
from regr.program.loss import NBCrossEntropyLoss, BCEWithLogitsIMLoss
from regr.program.metric import MacroAverageTracker, PRF1Tracker, MetricTracker, CMWithLogitsMetric, DatanodeCMMetric
import logging
from reader import read_data
from regr.graph import Graph, Concept, Relation, ifL, andL, notL, existsL
from regr.program.primaldualprogram import PrimalDualProgram
from regr.sensor.pytorch import ModuleLearner
from regr.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from regr.sensor.pytorch.sensors import ReaderSensor, JointSensor, FunctionalSensor
from utils import Generator, make_facts, label_reader, RobertaTokenizer, BBRobert,SimpleTokenizer
from regr.program import SolverPOIProgram, IMLProgram
from regr.program.model.pytorch import SolverModel, IMLModel

parser = argparse.ArgumentParser(description='Run beleifebank Main Learning Code')

parser.add_argument('--cuda', dest='cuda_number', default=0, help='cuda number to train the models on',type=int)
parser.add_argument('--epoch', dest='cur_epoch', default=1, help='number of epochs you want your model to train on',type=int)

parser.add_argument('--samplenum', dest='samplenum', default=40, help='sample sizes for low data regime 10,20,40 max 37',type=int)

parser.add_argument('--simple_model', dest='simple_model', default=False, help='use the simplet model',type=bool)
parser.add_argument('--pd', dest='primaldual', default=False, help='whether or not to use primaldual constriant learning',type=bool)
parser.add_argument('--iml', dest='IML', default=False, help='whether or not to use IML constriant learning',type=bool)
parser.add_argument('--sam', dest='SAM', default=False, help='whether or not to use sampling learning',type=bool)

parser.add_argument('--batch', dest='batch_size', default=4, help='batch size for neural network training',type=int)
parser.add_argument('--beta', dest='beta', default=0.1, help='primal dual or IML multiplier',type=float)
parser.add_argument('--lr', dest='learning_rate', default=2e-4, help='learning rate of the adam optimiser',type=float)

args = parser.parse_args()

logging.basicConfig(level=logging.INFO)


calibration_data,silver_data,constraints_yes,constraints_no=read_data(batch_size=args.batch_size,sample_size=args.samplenum)
train_size=len(calibration_data)*3//4
calibration_data_dev=calibration_data[train_size:]
calibration_data=calibration_data[:train_size]
cuda_number= args.cuda_number
device = "cuda:"+str(cuda_number) if torch.cuda.is_available() else 'cpu'

print("device is : ",device)

def guess_pair_yes(sentence, arg1, arg2):

    if len(sentence)<2 or arg1==arg2:
        return False
    sentence1, sentence2 = arg1.getAttribute('sentence'), arg2.getAttribute('sentence')
    if sentence1 in constraints_yes and sentence2 in constraints_yes[sentence1]:
        return True
    else:
        return False

def guess_pair_no(sentence, narg1, narg2):

    if len(sentence)<2 or narg1==narg2:
        return False
    sentence1, sentence2 = narg1.getAttribute('sentence'), narg2.getAttribute('sentence')
    if sentence1 in constraints_no and sentence2 in constraints_no[sentence1]:
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
    implication = Concept(name='implication')
    i_arg1, i_arg2 = implication.has_a(arg1=facts, arg2=facts)

    nimplication = Concept(name='nimplication')
    ni_arg1, ni_arg2 = nimplication.has_a(narg1=facts, narg2=facts)

    ifL(andL(fact_check('x'), existsL(implication('s', path=('x', implication)))), fact_check(path=('s', i_arg2)))
    #ifL(implication('s'), ifL(fact_check(path=('s',i_arg1.reversed)),fact_check(path=('s',i_arg2.reversed )) ) )
    #ifL(andL(implication('s'),fact_check(path=('s',i_arg1.reversed)) ,fact_check(path=('s',i_arg2.reversed )) ) )
    ifL(andL(fact_check('x'), existsL(nimplication('s', path=('x', nimplication)))), notL(fact_check(path=('s', ni_arg2))))

subject['name'] = ReaderSensor(keyword='name')
subject['facts'] = ReaderSensor(keyword='facts')
subject['labels'] = ReaderSensor(keyword='labels')

facts[subject_facts_contains,"name", "sentence", 'label'] = JointSensor(\
    subject['name'], subject['facts'], subject['labels'],forward=make_facts,device=device)
facts[fact_check] = FunctionalSensor(subject_facts_contains, "label", forward=label_reader, label=True,device=device)

implication[i_arg1.reversed, i_arg2.reversed] = CompositionCandidateSensor(facts['sentence'],relations=(i_arg1.reversed, i_arg2.reversed),forward=guess_pair_yes,device=device)
nimplication[ni_arg1.reversed, ni_arg2.reversed] = CompositionCandidateSensor(facts['sentence'],relations=(ni_arg1.reversed, ni_arg2.reversed),forward=guess_pair_no,device=device)

if not args.simple_model:
    facts["token_ids", "Mask"] = JointSensor("name", "sentence", forward=RobertaTokenizer(),device=device)
    facts[fact_check] = ModuleLearner("token_ids", "Mask", module=BBRobert(),device=device)
else:
    facts["emb"] = JointSensor("name", "sentence", forward=SimpleTokenizer(device),device=device)
    facts[fact_check] = ModuleLearner("emb", module=torch.nn.Linear(96, 2),device=device)

f=open("salam.txt","w")
if not args.primaldual and not args.IML and not args.SAM:
    program = SolverPOIProgram(graph, poi=[facts[fact_check],implication,nimplication],inferTypes=['ILP','local/argmax'],\
                    loss=MacroAverageTracker(NBCrossEntropyLoss()),metric={'ILP': PRF1Tracker(DatanodeCMMetric()),\
                                                'softmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},f=f)
elif args.primaldual:
    program = PrimalDualProgram(graph,SolverModel, poi=[facts[fact_check],implication,nimplication],inferTypes=['ILP','local/argmax'],\
                    loss=MacroAverageTracker(NBCrossEntropyLoss()),metric={'ILP': PRF1Tracker(DatanodeCMMetric()),\
                                               'softmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},beta=args.beta,device=device,f=f)
elif args.IML:
    program = IMLProgram(graph, poi=[facts[fact_check],implication,nimplication],inferTypes=['ILP','local/argmax'],\
                   loss=MacroAverageTracker(BCEWithLogitsIMLoss(lmbd=args.beta)),metric={'ILP': PRF1Tracker(DatanodeCMMetric()),\
                                               'softmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})
elif args.SAM:
    program = SampleLossProgram(graph, SolverModel,poi=[facts[fact_check],implication,nimplication],inferTypes=['ILP','local/argmax'],
        metric={'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},loss=MacroAverageTracker(NBCrossEntropyLoss()),sample=True,sampleSize=50,sampleGlobalLoss=True,beta=args.beta,device=device)

program.train(calibration_data,valid_set=calibration_data_dev,test_set=silver_data, train_epoch_num=args.cur_epoch, Optim=lambda param: AdamW(param, lr = args.learning_rate ,eps = 1e-9 ),device=device)

ac_, t_ = 0, 0
for datanode in program.populate(silver_data, device="cpu"):
    #     tdatanode = datanode.findDatanodes(select = context)[0]
    #     print(len(datanode.findDatanodes(select = context)))
    #     print(tdatanode.getChildDataNodes(conceptName=step))
    datanode.inferILPResults()
    verifyResult = datanode.verifyResultsLC()
    verifyResultILP = datanode.verifyResultsLC()
    ac_ += sum([verifyResultILP[lc]['satisfied'] for lc in verifyResultILP])
    t_ +=len(verifyResultILP.keys())

print("constraint accuracy: ", ac_ / t_ )

#, c_warmup_iters=0,test_set=silver_data
f.close()
_,silver_data_test,constraints_yes,constraints_no=read_data(batch_size=32*16,sample_size=40)

program.test(silver_data_test, device=device)

