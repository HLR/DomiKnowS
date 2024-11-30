import torch, logging
from transformers import AdamW

from utils import generate_test_case
from domiknows.graph.concept import EnumConcept
from domiknows.graph import Graph, Concept, Relation
from domiknows.program import SolverPOIProgram
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.metric import MacroAverageTracker, PRF1Tracker
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.sensor.pytorch.sensors import ReaderSensor, JointSensor, FunctionalSensor, FunctionalReaderSensor
from domiknows.program.lossprogram import LeftProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows.graph.LeftLogic import LeftLogicElement

data_samples=100
logging.basicConfig(level=logging.INFO)
Graph.clear()
Concept.clear()
Relation.clear()

with Graph(name='global') as graph:
    xcon = Concept(name='xcon')

    ycon1 = xcon(name='ycon1')
    ycon2 = xcon(name='ycon2')

    zcon = xcon(name='zcon', ConceptClass=EnumConcept, values=['1', '2', '3', '4'])

reader1 = [generate_test_case() for _ in range(data_samples)]
reader2 = [generate_test_case() for _ in range(data_samples)]

xcon["value1"] = ReaderSensor(keyword="value1")
xcon["value2"] = ReaderSensor(keyword="value2")

y1_module=torch.nn.Linear(5, 2)
y2_module=torch.nn.Linear(5, 2)

xcon[ycon1] = ModuleLearner("value1", module=y1_module)
xcon[ycon2] = ModuleLearner("value2", module=y2_module)
xcon[ycon1] = ReaderSensor(keyword="y1", label=True)
xcon[ycon2] = ReaderSensor(keyword="y2", label=True)


def text_maker(y1,y2):
    return [[str(y1.sigmoid().argmax().item()),str(y2.sigmoid().argmax().item())]]
xcon["text_tensorlistofys"] = FunctionalSensor(xcon[ycon1],xcon[ycon2], forward=text_maker)

def add_ys(t):
    return torch.FloatTensor([[float(t[0][0]),float(t[0][1])]])

xcon["tensorlistofys"] = FunctionalSensor("text_tensorlistofys", forward=add_ys)
xcon[zcon] = ModuleLearner("tensorlistofys", module=torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Linear(4, 4)))
xcon[zcon] = ReaderSensor(keyword="z", label=True)

program = SolverPOIProgram(graph,poi=[xcon[ycon1],xcon[ycon2]],inferTypes=['local/softmax'],
                           loss=MacroAverageTracker(NBCrossEntropyLoss()))
program.train(reader1, train_epoch_num=10, Optim=lambda param: AdamW(param, lr = 1e-3,eps = 1e-6 ))

#____________________________
# y1 = 1 , y2 = 1 , [0,0,1] sumL(y1,y2) = [p1,p2,p3]

sumL=LeftLogicElement(graph,"sumL")
multiplicationL=LeftLogicElement(graph,"multiplicationL") # y1= 0, y2= 0 -> 1 [1,0,0,0] y1= 1, y2= 1 -> 4 [0,0,0,1]
inverseL=LeftLogicElement(graph,"inverseL") # y1 =0 - > y1=1

# multiplicationL(inverseL(y1),y2)
# none learbale

def sumys(y1, y2):
    sum_result = y1 + y2
    output = torch.zeros(3, dtype=torch.int32)
    if sum_result == 0:
        output[0] = 1
    elif sum_result == 1:
        output[1] = 1
    else:
        output[2] = 1
    return output


program = LeftProgram(
    graph, SolverModel, poi=[xcon[zcon]],
    inferTypes=['local/argmax'],
    loss=MacroAverageTracker(NBCrossEntropyLoss()),
    Inferences=[sumys(data["y1"],data["y2"]) for data in reader2] ,
    LCList=[sumL(ycon1,ycon2) for _ in range(data_samples)],
    device='cpu'
)

program.train(reader2, train_epoch_num=10, Optim=lambda param: AdamW(param, lr = 8e-3,eps = 1e-6 ),device="cpu")