import torch
from transformers import AdamW
from regr.program.loss import NBCrossEntropyLoss, BCEWithLogitsIMLoss
from regr.program.metric import MacroAverageTracker, PRF1Tracker, MetricTracker, CMWithLogitsMetric, DatanodeCMMetric
import logging
from reader import read_data
from regr.graph import Graph, Concept, Relation, ifL
from regr.program.primaldualprogram import PrimalDualProgram
from regr.sensor.pytorch import ModuleLearner
from regr.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from regr.sensor.pytorch.sensors import ReaderSensor, JointSensor, FunctionalSensor
from utils import Generator, make_facts, label_reader, RobertaTokenizer, BBRobert
from regr.program import SolverPOIProgram, IMLProgram
from regr.program.model.pytorch import SolverModel, IMLModel

logging.basicConfig(level=logging.INFO)
calibration_data,silver_data,constraints=read_data(batch_size=32,sample_size=100000)
device='cuda:0'

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
    implication = Concept(name='implication')
    i_arg1, i_arg2 = implication.has_a(arg1=facts, arg2=facts)

    ifL(fact_check('x'), fact_check(path=('x', implication, i_arg2)))

subject['name'] = ReaderSensor(keyword='name')
subject['facts'] = ReaderSensor(keyword='facts')
subject['labels'] = ReaderSensor(keyword='labels')

facts[subject_facts_contains,"name", "sentence", 'label'] = JointSensor(\
    subject['name'], subject['facts'], subject['labels'],forward=make_facts)
facts[fact_check] = FunctionalSensor(subject_facts_contains, "label", forward=label_reader, label=True)

implication[i_arg1.reversed, i_arg2.reversed] = CompositionCandidateSensor(facts['sentence'],relations=(i_arg1.reversed, i_arg2.reversed),forward=guess_pair)

facts["token_ids", "Mask"] = JointSensor("name", "sentence", forward=RobertaTokenizer(),device=device)
facts[fact_check] = ModuleLearner("token_ids", "Mask", module=BBRobert())

#program = SolverPOIProgram(graph, poi=[facts[fact_check],implication],inferTypes=['ILP','local/argmax'],\
#                loss=MacroAverageTracker(NBCrossEntropyLoss()),metric={'ILP': PRF1Tracker(DatanodeCMMetric()),\
#                                            'softmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})

program = PrimalDualProgram(graph,SolverModel, poi=[facts[fact_check],implication],inferTypes=['ILP','local/argmax'],\
                loss=MacroAverageTracker(NBCrossEntropyLoss()),metric={'ILP': PRF1Tracker(DatanodeCMMetric()),\
                                            'softmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},beta=0.3)

#program = IMLProgram(graph, poi=[facts[fact_check],implication],inferTypes=['ILP','local/argmax'],\
 #               loss=MacroAverageTracker(BCEWithLogitsIMLoss(lmbd=0.5)),metric={'ILP': PRF1Tracker(DatanodeCMMetric()),\
 #                                           'softmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})
program.train(calibration_data,valid_set=silver_data, train_epoch_num=30, Optim=lambda param: AdamW(param, lr = 4e-6,eps = 1e-9 ), device=device)

#program.test(silver_data)