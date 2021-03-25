
import torch
import numpy as np
from transformers import  AdamW
from regr.graph.relation import disjoint
from regr.program import POIProgram
from torch import nn
from regr.program.metric import MacroAverageTracker, PRF1Tracker, MetricTracker, CMWithLogitsMetric
from regr.program.loss import NBCrossEntropyLoss
import logging
from regr.program.model.primaldual import PrimalDualModel
from regr.sensor.pytorch.learners import ModuleLearner
from regr.sensor.pytorch.sensors import ReaderSensor, JointSensor, FunctionalSensor, FunctionalReaderSensor
from regr.graph.logicalConstrain import nandL, ifL, V, orL, andL, existsL, notL, atLeastL, atMostL, eqL
from regr.graph import Graph, Concept, Relation
from preprocess import make_reader
from regr.sensor.pytorch.relation_sensors import EdgeSensor, CompositionCandidateReaderSensor, \
    CompositionCandidateSensor
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
cur_epoch=1
cur_device="auto"
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

    ifL(is_more, V(name='x'), is_less, V(name='y', v=('x', symmetric.name, s_arg2.name)))
    ifL(is_less, V(name='x'), is_more, V(name='y', v=('x', symmetric.name, s_arg2.name)))

    #ifL(is_more, V(name='x'), is_less, V(name='y', v=('x', eqL(symmetric, 'label_', {1}), s_arg2.name)))
    #ifL(is_less, V(name='x'), is_more, V(name='y', v=('x', eqL(symmetric, 'label_', {1}), s_arg2.name)))

    transitive = Concept(name='transitive')
    t_arg1, t_arg2, t_arg3 = transitive.has_a(arg11=question, arg22=question, arg33=question)

    ifL(andL(is_more, V(name='x'),is_more,V(name='z', v=('x', transitive.name, t_arg2.name))),
        is_more, V(name='y', v=('x', transitive.name, t_arg3.name)))

    ifL(andL(is_more, V(name='x'),is_less,V(name='z', v=('x', transitive.name, t_arg2.name))),
        is_less, V(name='y', v=('x', transitive.name, t_arg3.name)))

    #ifL(andL(is_more, V(name='x'),is_more,V(name='z', v=('x', eqL(transitive, 'label_', {1}), t_arg2.name))),
    #    is_more, V(name='y', v=('x', eqL(transitive, 'label_', {1}), t_arg3.name)))

    #ifL(andL(is_more, V(name='x'),is_less,V(name='z', v=('x', eqL(transitive, 'label_', {1}), t_arg2.name))),
    #    is_less, V(name='y', v=('x', eqL(transitive, 'label_', {1}), t_arg3.name)))

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


question[para_quest_contains, "question_paragraph", 'text', "is_more_", "is_less_", "no_effect_", "quest_id"] = JointSensor(
    paragraph['paragraph'], paragraph['question_list']
    , paragraph['less_list'], paragraph['more_list'], paragraph['no_effect_list'], paragraph['quest_ids'], forward=make_questions)

question["token_ids", "Mask"] = JointSensor(para_quest_contains, "question_paragraph", 'text', forward=RobertaTokenizer("more"))
#question["token_ids_more", "Mask_more"] = JointSensor(para_quest_contains, "question_paragraph", 'text', forward=RobertaTokenizer("more"))
#question["token_ids_less", "Mask_less"] = JointSensor(para_quest_contains, "question_paragraph", 'text', forward=RobertaTokenizer("less"))
#question["token_ids_no_effect", "Mask_no_effect"] = JointSensor(para_quest_contains, "question_paragraph", 'text', forward=RobertaTokenizer("no effect"))

def label_reader(_, label):
    return label

question[is_more] = FunctionalSensor(para_quest_contains, "is_more_", forward=label_reader, label=True)
question[is_less] = FunctionalSensor(para_quest_contains, "is_less_", forward=label_reader, label=True)
question[no_effect] = FunctionalSensor(para_quest_contains, "no_effect_", forward=label_reader, label=True)

roberta_model=DariusWIQA_Robert()

#question["all_embs"] = ModuleLearner("token_ids_more", "Mask_more","token_ids_less", "Mask_less","token_ids_no_effect", "Mask_no_effect", module=roberta_model)
class EmbSeprateor:
    def __init__(self,dim):
        self.dim=dim
    def __call__(self,_,all_embs):
        return all_embs[self.dim]
question["emb"] = ModuleLearner("token_ids", "Mask", module=roberta_model)
#question["emb_is_more"] = ModuleLearner("token_ids_more", "Mask_more", module=roberta_model)
#question["emb_is_less"] = ModuleLearner("token_ids_less", "Mask_less", module=roberta_model)
#question["emb_no_effect"] = ModuleLearner("token_ids_no_effect", "Mask_no_effect", module=roberta_model)

from preprocess import make_pair,make_pair_with_labels,make_triple,make_triple_with_labels,guess_pair,guess_triple

symmetric[s_arg1.reversed, s_arg2.reversed] = CompositionCandidateSensor(question['quest_id'], relations=(s_arg1.reversed, s_arg2.reversed), forward=guess_pair)
#symmetric['neighbor'] = DataNodeReaderSensor(s_arg1.reversed, s_arg2.reversed, keyword='links', forward=guess_pair_datanode_2)
#symmetric[s_arg1.reversed, s_arg2.reversed,"labels_"] = JointSensor(question['quest_id'], forward=make_pair)
#symmetric[s_arg1.reversed, s_arg2.reversed,"labels_"] = JointSensor(question['quest_id'], forward=make_pair_with_labels)

transitive[t_arg1.reversed, t_arg2.reversed, t_arg3.reversed] = CompositionCandidateSensor(question['quest_id'], relations=(t_arg1.reversed, t_arg2.reversed, t_arg3.reversed), forward=guess_triple)
#transitive[t_arg1.reversed, t_arg2.reversed, t_arg3.reversed,"labels_"] = JointSensor(question['quest_id'], forward=make_triple_with_labels)

class Classifier(torch.nn.Linear):
    def __init__(self, dim_in=roberta_model.last_layer_size):
        super().__init__(dim_in, 2)

#shared_layer = Classifier()



question[is_more] = ModuleLearner("emb", module=Classifier())
question[is_less] = ModuleLearner("emb", module=Classifier())
question[no_effect] = ModuleLearner("emb", module=Classifier())

class WIQAModel(PrimalDualModel):
    def __init__(self, graph,poi,loss,metric):
        super().__init__(
            graph,
            poi=poi,
            loss=loss,
            metric=metric)

class NBCrossEntropyLoss_(torch.nn.CrossEntropyLoss):
    def forward(self, input, target, *args, **kwargs):
        input = input.view(-1, input.shape[-1])
        target = target.view(-1).to(dtype=torch.long)
        #print(input.shape,target.shape,super().forward(input, target, *args, **kwargs))
        #print(input,target)
        return super().forward(input, target, *args, **kwargs)

#program = POIProgram(graph, loss=MacroAverageTracker(NBCrossEntropyLoss_()), metric=PRF1Tracker())
program = LearningBasedProgram(graph, model_helper(PoiModel,#WIQAModel
                    poi=[question[is_less], question[is_more], question[no_effect],symmetric,transitive],
                            loss=MacroAverageTracker(NBCrossEntropyLoss_()), metric=PRF1Tracker()))
logging.basicConfig(level=logging.INFO)
program.train(reader, train_epoch_num=cur_epoch, Optim=AdamW, device=cur_device)


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
counter=0
ac_=0
for paragraph_ in program.populate(reader, device=cur_device):
    #print("paragraph:", paragraph_.getAttribute('paragraph'))
    paragraph_.inferILPResults(is_more,is_less,no_effect,fun=None)
    questions_id,results=[],[]
    for question_ in paragraph_.getChildDataNodes():
        #print(question_.getAttribute('text'))
        #print(question_.getAttribute('question_paragraph'))
        #print(question_.getAttribute('emb'))
        questions_id.append(question_.getAttribute('quest_id'))
        results.append((question_.getAttribute(is_more,"ILP"),question_.getAttribute(is_less,"ILP"),question_.getAttribute(no_effect,"ILP")))

        #predict_is_more=question_.getAttribute(is_more).softmax(-1).argmax().item()
        predict_is_more_value=question_.getAttribute(is_more).softmax(-1)[0].item()
        predict_is_less_value=question_.getAttribute(is_less).softmax(-1)[0].item()
        predict_no_effect_value=question_.getAttribute(no_effect).softmax(-1)[0].item()
        counter+=1
        ac_+=np.array([predict_is_more_value,predict_is_less_value,predict_no_effect_value]).argmax()==np.array([question_.getAttribute("is_more_"),question_.getAttribute("is_less_"),question_.getAttribute("no_effect_")]).argmax()

    if not is_ILP_consistant(questions_id,results):
        print("ILP inconsistency")
print("dev accuracy:",ac_/counter)