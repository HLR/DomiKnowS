import torch
from regr.program.loss import NBCrossEntropyLoss, BCEWithLogitsIMLoss
from regr.program.metric import MacroAverageTracker, PRF1Tracker, MetricTracker, CMWithLogitsMetric, DatanodeCMMetric
import logging
from reader import read_data
from regr.graph import Graph, Concept, Relation, ifL
from regr.sensor.pytorch import ModuleLearner
from regr.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from regr.sensor.pytorch.sensors import ReaderSensor, JointSensor, FunctionalSensor
from utils import Generator
from regr.program import LearningBasedProgram, IMLProgram, SolverPOIProgram

calibration_data,silver_data,constraints=read_data()

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

    ifL(fact_check('x'), fact_check(path=('x', facts, i_arg2)))

subject['name'] = ReaderSensor(keyword='name')
subject['facts'] = ReaderSensor(keyword='facts')
subject['labels'] = ReaderSensor(keyword='labels')

def make_facts(name, facts, labels):
    return torch.ones((len(facts), 1)), [name for i in range(len(facts))],facts,labels

facts[subject_facts_contains,"name", "sentence", 'label'] = JointSensor(\
    subject['name'], subject['facts'], subject['labels'],forward=make_facts)



def label_reader(_, label):
    return {"no":0,"yes":1}.get(label)

# we load the is_more, is_less and no_effect attributes with label=True to tell the program
# about the real label of these questions for training
facts[fact_check] = FunctionalSensor(subject_facts_contains, "label", forward=label_reader, label=True)

def guess_pair(sentence, arg1, arg2):

    if len(sentence)<2 or arg1==arg2:
        return False
    sentence1, sentence2 = arg1.getAttribute('sentence'), arg2.getAttribute('sentence')
    if sentence2 in constraints[sentence1]:
        return True
    else:
        return False

implication[i_arg1.reversed, i_arg2.reversed] = CompositionCandidateSensor(facts['sentence'],relations=(i_arg1.reversed, i_arg2.reversed),forward=guess_pair)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("allenai/macaw-large")
model = AutoModelForSeq2SeqLM.from_pretrained("allenai/macaw-large")
facts[fact_check] = ModuleLearner("name", "facts", module=Generator(tokenizer,model))

program = SolverPOIProgram(graph, poi=[facts[fact_check],implication],loss=MacroAverageTracker(NBCrossEntropyLoss()), metric=PRF1Tracker())
program.test(calibration_data)