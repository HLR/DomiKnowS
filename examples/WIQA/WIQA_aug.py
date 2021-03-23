import torch
import numpy as np

from regr.graph.relation import disjoint
from regr.program import POIProgram
from torch import nn
from regr.program.metric import MacroAverageTracker, PRF1Tracker, MetricTracker, CMWithLogitsMetric
from regr.program.loss import NBCrossEntropyLoss
import logging
from regr.program.model.primaldual import PrimalDualModel
from regr.sensor.pytorch.learners import ModuleLearner
from regr.sensor.pytorch.sensors import ReaderSensor, JointSensor, FunctionalSensor, FunctionalReaderSensor
from regr.graph.logicalConstrain import nandL,ifL, V, orL, andL, existsL, notL, atLeastL, atMostL
from regr.graph import Graph, Concept, Relation
from preprocess import make_reader
from regr.sensor.pytorch.relation_sensors import EdgeSensor, CompositionCandidateReaderSensor
from regr.sensor.pytorch.query_sensor import DataNodeReaderSensor
from regr.program import LearningBasedProgram
from regr.program.model.pytorch import model_helper, PoiModel
from preprocess import RobertaTokenizer,DariusWIQA_Robert,is_ILP_consistant
def guess_pair_datanode_2(*_, data, datanode):
    quest1_node = datanode.relationLinks[s_arg1.name][0]
    quest2_node = datanode.relationLinks[s_arg2.name][0]
    quest1=quest1_node.getAttribute('quest_id')
    quest2=quest2_node.getAttribute('quest_id')
    if quest1 in quest2 and "_symmetric" in quest2: #directed?
        return True
    else:
        return False

reader = make_reader(file_address="data/WIQA_AUG/train.jsonl", sample_num=10)
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

    disjoint(is_more, is_less, no_effect)

    symmetric = Concept(name='symmetric')
    s_arg1, s_arg2 = symmetric.has_a(arg1=question, arg2=question)
    #ifL(symmetric, ('x', 'y'), orL( andL(is_more, 'x', is_less, 'y'), andL(is_less, 'x', is_more, 'y')))

    ifL(is_more, V(name='x'), is_less, V(name='y', v=('x', symmetric.name, s_arg2.name)))
    ifL(is_less, V(name='x'), is_more, V(name='y', v=('x', symmetric.name, s_arg2.name)))

    transitive = Concept(name='transitive')
    t_arg1, t_arg2, t_arg3 = transitive.has_a(arg1=question, arg2=question, arg3=question)
    #ifL(eqL(transitive , 'label', {1}), ('x', 'y','z'), orL(
    #   ifL( andL(is_more, 'x', is_more), 'y',is_more, 'z')),
    #   ifL( andL(is_more, 'x', is_less, 'y'),is_less, 'z')
    #)
    ifL(andL(is_more, V(name='x'),is_more,V(name='z', v=('x', transitive.name, t_arg2.name))),
        is_more, V(name='y', v=('x', transitive.name, t_arg3.name)))

    ifL(andL(is_more, V(name='x'),is_less,V(name='z', v=('x', transitive.name, t_arg2.name))),
        is_less, V(name='y', v=('x', transitive.name, t_arg3.name)))

print("Sensor part:")

paragraph['paragraph'] = ReaderSensor(keyword='paragraph')
paragraph['question_list'] = ReaderSensor(keyword='question_list')

paragraph['less_list'] = ReaderSensor(keyword='less_list')
paragraph['more_list'] = ReaderSensor(keyword='more_list')
paragraph['no_effect_list'] = ReaderSensor(keyword='no_effect_list')

paragraph['quest_ids'] = ReaderSensor(keyword='quest_ids')

def str_to_int_list(x):
    return torch.LongTensor([[int(i)] for i in x])
def make_questions(paragraph, question_list, less_list, more_list, no_effect_list, quest_ids):
    #print("make_questions", paragraph,question_list,less_list.split("@@"))
    return torch.ones((len(question_list.split("@@")), 1)), [paragraph for i in range(len(question_list.split("@@")))], \
           question_list.split("@@"), str_to_int_list(less_list.split("@@")), str_to_int_list(more_list.split("@@")),\
           str_to_int_list(no_effect_list.split("@@")), quest_ids.split("@@")


question[para_quest_contains, "question_paragraph", 'text', "is_more", "is_less", "no_effect", "quest_id"] = JointSensor(
    paragraph['paragraph'], paragraph['question_list']
    , paragraph['less_list'], paragraph['more_list'], paragraph['no_effect_list'], paragraph['quest_ids'], forward=make_questions)


question["token_ids_more", "Mask_more"] = JointSensor(para_quest_contains, "question_paragraph", 'text', forward=RobertaTokenizer("more"))
question["token_ids_less", "Mask_less"] = JointSensor(para_quest_contains, "question_paragraph", 'text', forward=RobertaTokenizer("less"))
question["token_ids_no_effect", "Mask_no_effect"] = JointSensor(para_quest_contains, "question_paragraph", 'text', forward=RobertaTokenizer("no effect"))

def label_reader(_, label):
    return label

question[is_more] = FunctionalSensor(para_quest_contains, "is_more", forward=label_reader, label=True)
question[is_less] = FunctionalSensor(para_quest_contains, "is_less", forward=label_reader, label=True)
question[no_effect] = FunctionalSensor(para_quest_contains, "no_effect", forward=label_reader, label=True)

roberta_model=DariusWIQA_Robert()

question["all_embs"] = ModuleLearner("token_ids_more", "Mask_more","token_ids_less", "Mask_less","token_ids_no_effect", "Mask_no_effect", module=roberta_model)
class EmbSeprateor:
    def __init__(self,dim):
        self.dim=dim
    def __call__(self,_,all_embs):
        return all_embs[self.dim]

question["emb_is_more"] = FunctionalSensor(para_quest_contains, "all_embs", forward=EmbSeprateor(0))
question["emb_is_less"] = FunctionalSensor(para_quest_contains, "all_embs", forward=EmbSeprateor(1))
question["emb_no_effect"] = FunctionalSensor(para_quest_contains, "all_embs", forward=EmbSeprateor(2))

from preprocess import make_pair,make_pair_with_labels,make_triple,make_triple_with_labels,guess_pair

#symmetric[s_arg1.reversed, s_arg2.reversed] = CompositionCandidateReaderSensor(question['quest_id'], keyword='links', relations=(s_arg1.reversed, s_arg2.reversed), forward=guess_pair)
#symmetric['neighbor'] = DataNodeReaderSensor(s_arg1.reversed, s_arg2.reversed, keyword='links', forward=guess_pair_datanode_2)
symmetric[s_arg1.reversed, s_arg2.reversed,"labels_"] = JointSensor(question['quest_id'], forward=make_pair)
#symmetric[s_arg1.reversed, s_arg2.reversed] = JointSensor(question['quest_id'], forward=make_pair_with_labels)

transitive[t_arg1.reversed, t_arg2.reversed, t_arg3.reversed,"labels_"] = JointSensor(question['quest_id'], forward=make_triple)

class Classifier(torch.nn.Linear):
    def __init__(self, dim_in=roberta_model.last_layer_size):
        super().__init__(dim_in, 2)

shared_layer = Classifier()



question[is_more] = ModuleLearner("emb_is_more", module=shared_layer)
question[is_less] = ModuleLearner("emb_is_less", module=shared_layer)
question[no_effect] = ModuleLearner("emb_no_effect", module=shared_layer)

class WIQAModel(PrimalDualModel):
    def __init__(self, graph,poi,loss,metric):
        super().__init__(
            graph,
            poi=poi,
            loss=loss,
            metric=metric)

#program = POIProgram(graph, loss=MacroAverageTracker(NBCrossEntropyLoss()), metric=PRF1Tracker())
program = LearningBasedProgram(graph, model_helper(PoiModel,#WIQAModel
                    poi=[question[is_less], question[is_more], question[no_effect],symmetric,transitive],
                            loss=MacroAverageTracker(NBCrossEntropyLoss()), metric=PRF1Tracker()))
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

for paragraph_ in program.populate(reader, device='auto'):
    #print("paragraph:", paragraph_.getAttribute('paragraph'))
    paragraph_.inferILPResults(is_more,is_less,no_effect,fun=None)
    questions_id,results=[],[]
    for question_ in paragraph_.getChildDataNodes():
        #print(question_.getAttribute('text'))
        #print(question_.getAttribute('question_paragraph'))
        #print(question_.getAttribute('quest_id'))
        questions_id.append(question_.getAttribute('quest_id'))
        results.append((question_.getAttribute(is_more,"ILP"),question_.getAttribute(is_less,"ILP"),question_.getAttribute(no_effect,"ILP")))
        #print(question_.getAttribute(is_more,"ILP"))
    if not is_ILP_consistant(questions_id,results):
        print("ILP inconsistency")
    #print("\nILP results for paragraph - %s"%(paragraph_.collectInferedResults(is_more, "ILP")))
    #print("_" * 20)
