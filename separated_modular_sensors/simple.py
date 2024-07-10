import torch, logging
from transformers import AdamW

from utils import generate_test_case
from domiknows.graph.concept import EnumConcept
from domiknows.graph import Graph, Concept, Relation
from domiknows.program import SolverPOIProgram
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.metric import MacroAverageTracker, PRF1Tracker
from domiknows.sensor.pytorch.sensors import ReaderSensor
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.sensor.pytorch.sensors import ReaderSensor, JointSensor, FunctionalSensor, FunctionalReaderSensor

logging.basicConfig(level=logging.INFO)
Graph.clear()
Concept.clear()
Relation.clear()

with Graph(name='global') as graph:
    xcon = Concept(name='xcon')

    ycon1 = xcon(name='ycon1')
    ycon2 = xcon(name='ycon2')

reader = [generate_test_case() for i in range(2)]

xcon["value"] = ReaderSensor(keyword="value")

xcon[ycon1] = ModuleLearner("value", module=torch.nn.Linear(5, 2))
xcon[ycon2] = ModuleLearner("value", module=torch.nn.Linear(5, 2))
xcon[ycon1] = ReaderSensor(keyword="y1", label=True)
xcon[ycon2] = ReaderSensor(keyword="y2", label=True)

program = SolverPOIProgram(graph,poi=[xcon[ycon1]],inferTypes=['local/softmax'], loss=MacroAverageTracker(NBCrossEntropyLoss()))
program.train(reader, train_epoch_num=3, Optim=lambda param: AdamW(param, lr = 1e-2,eps = 1e-5 ))

program = SolverPOIProgram(graph,poi=[xcon[ycon2]],inferTypes=['local/softmax'], loss=MacroAverageTracker(NBCrossEntropyLoss()))
program.train(reader, train_epoch_num=3, Optim=lambda param: AdamW(param, lr = 1e-2,eps = 1e-5 ))
