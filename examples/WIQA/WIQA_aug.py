import torch
import numpy as np
from regr.program import POIProgram
from torch import nn
from regr.program.metric import MacroAverageTracker, PRF1Tracker, MetricTracker, CMWithLogitsMetric
from regr.program.loss import NBCrossEntropyLoss
import logging
from regr.sensor.pytorch.learners import ModuleLearner
from regr.sensor.pytorch.sensors import ReaderSensor, JointSensor, FunctionalSensor, FunctionalReaderSensor
from regr.graph.logicalConstrain import nandL,ifL, V, orL, andL, existsL, notL, atLeastL, atMostL
from regr.graph import Graph, Concept, Relation
from preprocess import make_reader

reader = make_reader(file_address="data/WIQA_AUG/train.jsonl", sample_num=100)
# reader.append({"paragraph":para,"more_list":more_list,"less_list":less_list,"no_effect_list":no_effect_list,"question_list":question_list})
# print(reader[0]["paragraph"])
# print(reader[1])
# exit()
Graph.clear()
Concept.clear()
Relation.clear()

with Graph('WIQA_graph') as graph:
    paragraph = Concept(name='paragraph')
    question = Concept(name='question')
    para_quest_contains, = paragraph.contains(question)
    is_more = question(name='is_more')
    is_less = question(name='is_less')
    no_effect = question(name='no_effect')

    nandL(is_more, is_less, no_effect)

    symmetric = Concept(name='symmetric')
    s_arg1, s_arg2 = symmetric.has_a(arg1=question, arg2=question)
    #ifL(symmetric, ('x', 'y'), orL( andL(is_more, 'x', is_less, 'y'), andL(is_less, 'x', is_more, 'y')))

    #ifL(is_more, V(name='x'), is_less, V(name='y', v=('x', symmetric.name, s_arg2.name)))

    transitive = Concept(name='transitive')
    t_arg1, t_arg2, t_arg3 = transitive.has_a(arg1=question, arg2=question, arg3=question)
    #ifL(eqL(transitive , 'label', {1}), ('x', 'y','z'), orL(
    #   ifL( andL(is_more, 'x', is_more), 'y',is_more, 'z')),
    #   ifL( andL(is_more, 'x', is_less, 'y'),is_less, 'z')
    #)

print("Sensor part:")

paragraph['paragraph'] = ReaderSensor(keyword='paragraph')
paragraph['question_list'] = ReaderSensor(keyword='question_list')

paragraph['less_list'] = ReaderSensor(keyword='less_list')
paragraph['more_list'] = ReaderSensor(keyword='more_list')
paragraph['no_effect_list'] = ReaderSensor(keyword='no_effect_list')

paragraph['quest_ids'] = ReaderSensor(keyword='quest_ids')


def make_questions(paragraph, question_list, less_list, more_list, no_effect_list, quest_ids):
    #print("make_questions", paragraph,question_list[0],less_list[0])
    return torch.ones((len(question_list[0]), 1)), [paragraph for i in range(
        len(question_list[0]))], question_list[0], less_list[0], more_list[0], no_effect_list[0], quest_ids[0]


question[para_quest_contains, "question_paragraph", 'text', "is_more", "is_less", "no_effect", "quest_id"] = JointSensor(
    paragraph['paragraph'], paragraph['question_list']
    , paragraph['less_list'], paragraph['more_list'], paragraph['no_effect_list'], paragraph['quest_ids'], forward=make_questions)

from preprocess import RobertaTokenizer,DariusWIQA_Robert
question["token_ids_more", "Mask_more"] = JointSensor(para_quest_contains, "question_paragraph", 'text', forward=RobertaTokenizer("more"))
question["token_ids_less", "Mask_less"] = JointSensor(para_quest_contains, "question_paragraph", 'text', forward=RobertaTokenizer("less"))
question["token_ids_no_effect", "Mask_no_effect"] = JointSensor(para_quest_contains, "question_paragraph", 'text', forward=RobertaTokenizer("no effect"))

def label_reader(_, label):
    return label

question[is_more] = FunctionalSensor(para_quest_contains, "is_more", forward=label_reader, label=True)
question[is_less] = FunctionalSensor(para_quest_contains, "is_less", forward=label_reader, label=True)
question[no_effect] = FunctionalSensor(para_quest_contains, "no_effect", forward=label_reader, label=True)

roberta_model=DariusWIQA_Robert()

question["emb_is_more"] = ModuleLearner("token_ids_more", "Mask_more", module=roberta_model)
question["emb_is_less"] = ModuleLearner("token_ids_less", "Mask_less", module=roberta_model)
question["emb_no_effect"] = ModuleLearner("token_ids_no_effect", "Mask_no_effect", module=roberta_model)

from preprocess import make_pair,make_pair_with_labels,make_triple,make_triple_with_labels

symmetric[s_arg1.reversed, s_arg2.reversed] = JointSensor(question['quest_id'], forward=make_pair)
transitive[t_arg1.reversed, t_arg2.reversed, t_arg3.reversed] = JointSensor(question['quest_id'], forward=make_triple)

class Classifier(torch.nn.Linear):
    def __init__(self, dim_in=roberta_model.last_layer_size):
        super().__init__(dim_in, 2)

shared_layer = Classifier()



question[is_more] = ModuleLearner("emb_is_more", module=shared_layer)
question[is_less] = ModuleLearner("emb_is_less", module=shared_layer)
question[no_effect] = ModuleLearner("emb_no_effect", module=shared_layer)

program = POIProgram(graph, loss=MacroAverageTracker(NBCrossEntropyLoss()), metric=PRF1Tracker())

logging.basicConfig(level=logging.INFO)
program.train(reader, train_epoch_num=1, Optim=torch.optim.Adam, device='auto')

print('Training result:')
print(program.model.loss)
print(program.model.metric)

print('-' * 40)

#
# test
#
# program.test(reader, device='auto')
# print('Testing result:')
# print(program.model.loss)
# print(program.model.metric)

print('-' * 40)

for node in program.populate(reader, device='auto'):
    print("paragraph:", node.getAttribute('paragraph'))
    for word_node in node.getChildDataNodes():
        print(word_node.getAttribute('text'))
        print(word_node.getAttribute('question_paragraph'))
    print("_" * 20)
