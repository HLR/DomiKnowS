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
#from regr.program.primaldualprogram import PrimalDualProgram
from regr.program.lossprogram import PrimalDualProgram
from regr.sensor.pytorch import ModuleLearner
from regr.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from regr.sensor.pytorch.sensors import ReaderSensor, JointSensor, FunctionalSensor
from utils import make_questions, label_reader, RobertaTokenizer, BBRobert,SimpleTokenizer
from regr.program import SolverPOIProgram, IMLProgram
from regr.program.model.pytorch import SolverModel, IMLModel

parser = argparse.ArgumentParser(description='Run beleifebank Main Learning Code')
parser.add_argument('--namesave', dest='namesave', default="modelname", help='model name to save', type=str)
parser.add_argument('--cuda', dest='cuda_number', default=0, help='cuda number to train the models on',type=int)
parser.add_argument('--epoch', dest='cur_epoch', default=4, help='number of epochs you want your model to train on',type=int)

parser.add_argument('--samplenum', dest='samplenum', default=2000, help='sample sizes for low data regime 10,20,40 max 37',type=int)

parser.add_argument('--simple_model', dest='simple_model', default=True, help='use the simplet model',type=bool)
parser.add_argument('--pd', dest='primaldual', default=False, help='whether or not to use primaldual constriant learning',type=bool)
parser.add_argument('--iml', dest='IML', default=False, help='whether or not to use IML constriant learning',type=bool)
parser.add_argument('--sam', dest='SAM', default=False, help='whether or not to use sampling learning',type=bool)

parser.add_argument('--batch', dest='batch_size', default=128, help='batch size for neural network training',type=int)
parser.add_argument('--beta', dest='beta', default=0.1, help='primal dual or IML multiplier',type=float)
parser.add_argument('--lr', dest='learning_rate', default=1e-5, help='learning rate of the adam optimiser',type=float)

args = parser.parse_args()

logging.basicConfig(level=logging.INFO)


train_data=read_data(sample_size=args.samplenum)
dev_data=read_data(sample_size=args.samplenum)
test_data=read_data(sample_size=args.samplenum)

cuda_number= args.cuda_number
device = "cuda:"+str(cuda_number) if torch.cuda.is_available() else 'cpu'

print("device is : ",device)

def guess_pair_yes(proof,strategy, arg1, arg2):

    if len(proof)<2 or arg1==arg2:
        return False
    proof1, proof2 = arg1.getAttribute('proof'), arg2.getAttribute('proof')
    strategy1, strategy2 = arg1.getAttribute('strategy'), arg2.getAttribute('strategy')
 #   if not (strategy1, strategy2) == ('proof', 'proof'):
  #      return False
    for i in proof1.split("OR"):
        for j in proof2.split("OR"):
            if i[0:i.rfind("->")].strip("()[]") in j:
                #print("True here",proof1,proof2)
                return True

    return False

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('belief_bank') as graph:
    Context = Concept(name='context')
    Question = Concept(name='question')
    context_question_contains, = Context.contains(Question)

    qlabel = Question(name='qlabel')
    implication = Concept(name='implication')
    i_arg1, i_arg2 = implication.has_a(arg1=Question, arg2=Question)

    ifL(andL(qlabel('x'), existsL(implication('s', path=('x', implication)))), qlabel(path=('s', i_arg2)))


Context['text'] = ReaderSensor(keyword='context')
Context['questionlist'] = ReaderSensor(keyword='questionslist')
Context['labellist'] = ReaderSensor(keyword='labelslist')
Context['proofslist'] = ReaderSensor(keyword='proofslist')
Context['strategieslist'] = ReaderSensor(keyword='strategieslist')

Question[context_question_contains,"paragraph", "text", 'label',"proof","strategy"] = JointSensor(\
    Context['text'], Context['questionlist'], Context['labellist'],Context['proofslist'],Context['strategieslist'],forward=make_questions,device=device)

Question[qlabel] = FunctionalSensor(context_question_contains, "label", forward=label_reader, label=True,device=device)

implication[i_arg1.reversed, i_arg2.reversed] = CompositionCandidateSensor(Question['proof'],Question['strategy'],relations=(i_arg1.reversed, i_arg2.reversed),forward=guess_pair_yes,device=device)

if not args.simple_model:
    Question["token_ids", "Mask"] = JointSensor("paragraph", "text", forward=RobertaTokenizer(),device=device)
    Question[qlabel] = ModuleLearner("token_ids", "Mask", module=BBRobert(),device=device)
else:
    Question["emb"] = JointSensor("paragraph", "text", forward=SimpleTokenizer(device),device=device)
    Question[qlabel] = ModuleLearner("emb", module=torch.nn.Linear(192, 2),device=device)

f=open("salam.txt","w")
if not args.primaldual and not args.IML and not args.SAM:
    program = SolverPOIProgram(graph, poi=[Question[qlabel],implication],inferTypes=['ILP','local/argmax'],\
                    loss=MacroAverageTracker(NBCrossEntropyLoss()),metric={'ILP': PRF1Tracker(DatanodeCMMetric()),\
                                                'softmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},f=f)
elif args.primaldual:
    program = PrimalDualProgram(graph,SolverModel, poi=[Question[qlabel],implication],inferTypes=['ILP','local/argmax'],\
                    loss=MacroAverageTracker(NBCrossEntropyLoss()),metric={'ILP': PRF1Tracker(DatanodeCMMetric()),\
                                               'softmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},beta=args.beta,device=device,f=f)
elif args.SAM:
    program = SampleLossProgram(graph, SolverModel,poi=[Question[qlabel],implication],inferTypes=['ILP','local/argmax'],
        metric={'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},loss=MacroAverageTracker(NBCrossEntropyLoss()),sample=True,sampleSize=50,sampleGlobalLoss=False,beta=args.beta,device=device)

for i in range(args.cur_epoch):
    program.train(train_data,valid_set=dev_data, train_epoch_num=1, Optim=lambda param: AdamW(param, lr = args.learning_rate ,eps = 1e-9 ),device=device)
    program.save(args.namesave + "_" + str(i))
