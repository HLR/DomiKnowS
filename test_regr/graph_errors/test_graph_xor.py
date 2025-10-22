import sys
sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('./')
sys.path.append('../')
import torch, random
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.metric import MacroAverageTracker, PRF1Tracker
from domiknows.sensor.pytorch.learners import ModuleLearner, TorchLearner
from domiknows.sensor.pytorch.sensors import ReaderSensor, JointSensor, FunctionalSensor, FunctionalReaderSensor, LabelReaderSensor
from domiknows.sensor.pytorch.relation_sensors import EdgeSensor
from domiknows.graph.logicalConstrain import nandL, ifL, V, orL, andL, existsL, notL, atLeastL, atMostL, eqL, xorL, exactL
from domiknows.graph import Graph, Concept, Relation, EnumConcept
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from domiknows.program import LearningBasedProgram, IMLProgram, SolverPOIProgram

with Graph('email_spam_consistency') as graph:
    email = Concept(name='email')

    m1 = email(name='model1_pred', ConceptClass=EnumConcept, values=['spam', 'not_spam'])
    m2 = email(name='model2_pred', ConceptClass=EnumConcept, values=['spam', 'not_spam'])

    xorL(m1.spam, m2.not_spam)

def random_cifar10_instance():
    email = [1]
    m1 = [random.randint(0,1)]
    m2 = [random.randint(0, 1)]

    data = {
        "email_id": [i for i in range(len(email))],
        "m1_id":  [i for i in range(len(m1))],
        "m2_id":  [i for i in range(len(m2))],
        "email": email,
        "m1": m1,
        "m2": m2,
    }
    return data

class DummyLearner(TorchLearner):

    def __init__(self, *pre, output_size=2):
        super(DummyLearner, self).__init__(*pre)
        self.output_size = output_size

    def forward(self, x):
        result = torch.zeros(len(x), self.output_size)
        random_indices = torch.randint(0, self.output_size, (len(x),))
        result[torch.arange(len(x)), random_indices] = 1
        return result

dataset = [random_cifar10_instance() for _ in range(1)]

email['email_id'] = ReaderSensor(keyword='email_id')
m1['m1_id'] = ReaderSensor(keyword='m1_id')
m2['m2_id'] = ReaderSensor(keyword='m2_id')

m1['m1'] = LabelReaderSensor(keyword='m1')
m2['m2'] = LabelReaderSensor(keyword='m2')

email[m1] = DummyLearner('email_id', output_size=2)
email[m2] = DummyLearner('email_id', output_size=2)

program = SolverPOIProgram(graph,poi=[email,m1,m2],inferTypes=['local/argmax'],loss=MacroAverageTracker(NBCrossEntropyLoss()),metric=PRF1Tracker())

def test_xor():
    for datanode in program.populate(dataset=dataset):
        datanode.inferILPResults()

        print(f"m1 :",datanode.getAttribute(m1,"local","argmax"))
        print(f"m1 ILP:", datanode.getAttribute(m1, "ILP"))

        print(f"m2 :",datanode.getAttribute(m2,"local","argmax"))
        print(f"m2 ILP:", datanode.getAttribute(m2, "ILP"))