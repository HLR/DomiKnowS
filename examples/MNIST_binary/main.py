import sys

sys.path.append('.')
sys.path.append('../..')
from regr.program.lossprogram import SampleLossProgram
from regr.program.model.pytorch import SolverModel
from regr.program.primaldualprogram import PrimalDualProgram

from regr.program import SolverPOIProgram, IMLProgram
from regr.sensor.pytorch.sensors import ReaderSensor, JointSensor, FunctionalSensor
from regr.sensor.pytorch.learners import ModuleLearner
from reader import create_readers
import torch.nn as nn
from regr.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric
from regr.program.loss import NBCrossEntropyLoss, BCEWithLogitsIMLoss
from graph import graph, image_group_contains,image,image_group,Zero,One,Two,Three,Four,Five,Six,Seven,Eight,Nine
import torchvision.datasets as datasets
import torch,argparse,torchvision
from model import MNISTCNN, MNISTLinear
import logging
parser = argparse.ArgumentParser(description='Run MNIST Binary Main Learning Code')
parser.add_argument('--namesave', dest='namesave', default="modelname", help='model name to save',type=str)


parser.add_argument('--cuda', dest='cuda_number', default=0, help='cuda number to train the models on',type=int)
parser.add_argument('--epoch', dest='cur_epoch', default=1, help='number of epochs you want your model to train on',type=int)
parser.add_argument('--lr', dest='learning_rate', default=2e-3, help='learning rate of the adam optimiser',type=float)
parser.add_argument('--ilp', dest='ilp', default=False, help='whether or not to use ilp',type=bool)
parser.add_argument('--pd', dest='primaldual', default=False, help='whether or not to use primaldual constriant learning',type=bool)
parser.add_argument('--iml', dest='IML', default=False, help='whether or not to use IML constriant learning',type=bool)
parser.add_argument('--sam', dest='SAM', default=False, help='whether or not to use sampling learning',type=bool)
parser.add_argument('--test', dest='test', default=True, help='dont train just test',type=bool)
parser.add_argument('--simple_model', dest='simple_model', default=False, help='use a simple baseline',type=bool)

parser.add_argument('--samplenum', dest='samplenum', default=99999, help='number of samples to train the model on',type=int)
parser.add_argument('--batch', dest='batch_size', default=30, help='batch size for neural network training',type=int)
parser.add_argument('--beta', dest='beta', default=0.005, help='primal dual or IML multiplier',type=float)
args = parser.parse_args()

from regr.utils import setProductionLogMode

productionMode = False
if productionMode:
    setProductionLogMode(no_UseTimeLog=False)

logging.basicConfig(level=logging.INFO)
transform_mnist=transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,), (0.3081,))])

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)

batch_size=args.batch_size
if args.test:
    batch_size*=10
mnist_trainset_reader=create_readers(mnist_trainset,args.samplenum,)
mnist_testset_reader=create_readers(mnist_testset,99999999,args.batch_size)

cuda_number= args.cuda_number
device = "cuda:"+str(cuda_number) if torch.cuda.is_available() else 'cpu'

print("device is : ",device)

image_group['pixels_group'] = ReaderSensor(keyword='pixels', device=device)
Numbers = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", \
           5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}

for i in range(10):
    image_group[Numbers[i]+'_group'] = ReaderSensor(keyword=Numbers[i], device=device)

def make_images(pixels_group):
    return torch.ones((torch.squeeze(pixels_group, 0).shape[0], 1)), torch.squeeze(pixels_group, 0)

def make_labels(pixels_group):
    return torch.ones((len(pixels_group), 1)) , torch.LongTensor([int(i) for i in pixels_group])

image[image_group_contains, "pixels"] = JointSensor(image_group['pixels_group'],forward=make_images)

for i in range(10):
    image[image_group_contains, Numbers[i]] = JointSensor(image_group[Numbers[i]+'_group'],forward=make_labels)

labels=[Zero,One,Two,Three,Four,Five,Six,Seven,Eight,Nine]

def label_reader(_, label):
    return label

for number,i in enumerate(labels):
    image[i] = FunctionalSensor(image_group_contains, Numbers[number], forward=label_reader, label=True)

for number, i in enumerate(labels):
    if args.simple_model:
        new_model=MNISTLinear()
    else:
        new_model=MNISTCNN((1, 28, 28),2,number)
    image[i] = ModuleLearner('pixels', module=new_model)

print("POI")
program = SolverPOIProgram(graph,poi=[image_group,image[Zero],image[One],image[Two],image[Three],image[Four],image[Five],image[Six],image[Seven],image[Eight],image[Nine]],inferTypes=['local/argmax'], loss=MacroAverageTracker(NBCrossEntropyLoss())\
                       ,metric={'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})

if args.ilp:
    print("ILP")
    program = SolverPOIProgram(graph,poi=[image_group,image[Zero],image[One],image[Two],image[Three],image[Four],image[Five],image[Six],image[Seven],image[Eight],image[Nine]],inferTypes=['ILP', 'local/argmax'], loss=MacroAverageTracker(NBCrossEntropyLoss())\
                           ,metric={'ILP': PRF1Tracker(DatanodeCMMetric('ILP')),'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})

if args.primaldual:
    print("PD")
    program = PrimalDualProgram(graph,SolverModel,poi=[image_group,image[Zero],image[One],image[Two],image[Three],image[Four],image[Five],image[Six],image[Seven],image[Eight],image[Nine]],inferTypes=['local/argmax'],\
                    loss=MacroAverageTracker(NBCrossEntropyLoss()),metric={#'ILP': PRF1Tracker(DatanodeCMMetric()),\
                                                'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},beta=args.beta,device=device)
if args.primaldual and args.ilp:
    print("PD and ILP")
    program = PrimalDualProgram(graph,SolverModel,poi=[image_group,image[Zero],image[One],image[Two],image[Three],image[Four],image[Five],image[Six],image[Seven],image[Eight],image[Nine]],inferTypes=['ILP','local/argmax'],\
                    loss=MacroAverageTracker(NBCrossEntropyLoss()),metric={'ILP': PRF1Tracker(DatanodeCMMetric()),\
                                                'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},beta=args.beta,device=device)
if args.IML:
    print("IML")
    program = IMLProgram(graph, poi=[image_group,image[Zero],image[One],image[Two],image[Three],image[Four],image[Five],image[Six],image[Seven],image[Eight],image[Nine]],\
                                    loss=MacroAverageTracker(BCEWithLogitsIMLoss(lmbd=args.beta)), metric=PRF1Tracker(),device=device)

if args.SAM:
    print("SAM")
    program = SampleLossProgram(graph, SolverModel, poi=[image_group,image[Zero],image[One],image[Two],image[Three],image[Four],image[Five],image[Six],image[Seven],image[Eight],image[Nine]],
                                inferTypes=['local/argmax'],
                                metric={'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},
                                loss=MacroAverageTracker(NBCrossEntropyLoss()), sample=True, sampleSize=250,
                                sampleGlobalLoss=True,device=device,beta=args.beta)

if args.SAM and args.ilp:
    print("SAM  and ILP")
    program = SampleLossProgram(graph, SolverModel, poi=[image_group,image[Zero],image[One],image[Two],image[Three],image[Four],image[Five],image[Six],image[Seven],image[Eight],image[Nine]],
                                inferTypes=['ILP','local/argmax'],
                                metric={'ILP': PRF1Tracker(DatanodeCMMetric()), 'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},
                                loss=MacroAverageTracker(NBCrossEntropyLoss()), sample=True, sampleSize=250,
                                sampleGlobalLoss=False,device=device,beta=args.beta)

for i in range(args.cur_epoch):
    if args.test:
        i=2
        program.load(args.namesave + "_" + str(i))
        program.test(mnist_testset_reader,device=device)

    else:
        program.train(mnist_trainset_reader,valid_set=mnist_testset_reader, train_epoch_num=1, Optim=lambda param: torch.optim.Adam(param, lr=args.learning_rate),device=device)
        program.save(args.namesave+"_"+str(i))
    import numpy as np

    ac_, t_ = 0, 0
    for datanode in program.populate(mnist_testset_reader, device="cpu"):
        datanode.inferILPResults()
        for i in datanode.getChildDataNodes():
            ilp_list=([i.getAttribute(*j) for j in [('<threehandwriting>', 'ILP'),\
                                                   ('<onehandwriting>', 'ILP'),\
                                                   ('<sevenhandwriting>' ,'ILP'),\
                                                   ('<sixhandwriting>', 'ILP'),\
                                                   ('<fivehandwriting>', 'ILP'),\
                                                   ('<ninehandwriting>', 'ILP'),\
                                                   ('<fourhandwriting>', 'ILP'),\
                                                   ('<eighthandwriting>', 'ILP'),\
                                                   ('<twohandwriting>', 'ILP'),\
                                                   ('<zerohandwriting>', 'ILP')]])
            if None in ilp_list:
                print("error None in Ilp results")
            elif not sum([j.item() for j in ilp_list])==1:
                print("exactL is not working")

        verifyResult = datanode.verifyResultsLC()
        verifyResultILP = datanode.verifyResultsLC()
        verify_vector = np.sum([verifyResultILP[lc]['verifyList'] for lc in verifyResultILP], axis=0)
        ac_ += np.sum(verify_vector)
        t_ += verify_vector.shape[0]

    print("constraint accuracy: ",ac_ / t_ * 100)
    if args.test:
        break

