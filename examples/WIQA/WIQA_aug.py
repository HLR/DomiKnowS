
import wget
import os
import torch
import numpy as np
from regr.program import POIProgram
from regr.graph import ifL, notL, andL, orL, nandL
from torch import nn
from regr.program.metric import MacroAverageTracker, PRF1Tracker, MetricTracker, CMWithLogitsMetric
from regr.program.loss import NBCrossEntropyLoss

from regr.sensor.pytorch.learners import ModuleLearner
from regr.sensor.pytorch.sensors import ReaderSensor, TorchEdgeSensor, JointSensor, FunctionalSensor, \
    FunctionalReaderSensor
import pandas as pd
sample_num=10000
counter=0
reader=[]
from collections import defaultdict
para_quest_dict=defaultdict(list)
para_quest_ans_dict=defaultdict(list)
quest_quest_dict=defaultdict(set)
data = pd.read_json("data/WIQA_AUG/train.jsonl", orient="records", lines=True)

for i, row in data.iterrows():
    datum={}

    if row["question"]["answer_label"].strip()=='more':
        datum["more"]=[1]
        datum["less"]=[0]
        datum["no_effect"]=[0]
    elif row["question"]["answer_label"].strip()=='less':
        datum["more"]=[0]
        datum["less"]=[1]
        datum["no_effect"]=[0]
    else:
        datum["more"]=[0]
        datum["less"]=[0]
        datum["no_effect"]=[1]

    para = " ".join([p.strip() for p in row["question"]["para_steps"] if len(p) > 0])
    paragraph_list=[p.strip() for p in row["question"]["para_steps"] if len(p) > 0]

    datum["paragraph"]=para
    datum["paragraph_list"]=paragraph_list

    question = row["question"]["stem"].strip()
    datum["question"]=question
    datum["ques_id"]=row["metadata"]["ques_id"]
    reader.append(datum)
    para_quest_dict[para].append(question)
    para_quest_ans_dict[para].append(row["question"]["answer_label"].strip())

    if not "@" in row["metadata"]["ques_id"]:
        quest_quest_dict[row["metadata"]["ques_id"]].add(row["metadata"]["ques_id"])
    else:
        if "_symmetric" in row["metadata"]["ques_id"]:
            quest_quest_dict[row["metadata"]["ques_id"].split("_symmetric")[0]].add(row["metadata"]["ques_id"])
        else:
            quest_quest_dict[row["metadata"]["ques_id"].split("@")[0]].add(row["metadata"]["ques_id"])
            quest_quest_dict[row["metadata"]["ques_id"].split("@")[1].split("_transit")[0]].add(row["metadata"]["ques_id"])

    counter+=1
    if counter>sample_num:
        break

from util_wiqa import make_graph_darius
for i in list(para_quest_dict.items()):
    if len(i[1]) >= 30:
        continue
    make_graph_darius(i[0],i[1],para_quest_ans_dict[i[0]])
    exit()
exit()

print(len(para_quest_dict),[len(i) for i in para_quest_dict.values()])
import seaborn as sns
from matplotlib import pyplot as plt
sns.displot(x=[len(i) for i in para_quest_dict.values()])
plt.figure(num=1)

sns.displot(x=[len(i) for i in quest_quest_dict.values()],binwidth=1)
plt.show()

from regr.graph import Graph, Concept, Relation

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('QA_graph') as graph:
    paragraph_question = Concept(name='paragraph_question')
    is_more = paragraph_question(name='is_more')
    is_less = paragraph_question(name='is_less')
    no_effect = paragraph_question(name='no_effect')

    nandL(is_more, is_less, no_effect)

    symmetric = Concept(name='symmetric')
    s_arg1,s_arg2=symmetric.has_a(arg1=paragraph_question, arg2=paragraph_question)
    ifL(symmetric, ('x', 'y'), orL( andL(is_more, 'x', is_less, 'y'), andL(is_less, 'x', is_more, 'y')))

    transitive = Concept(name='transitive')
    t_arg1,t_arg2,t_arg3 =transitive.has_a(arg1=paragraph_question, arg2=paragraph_question, arg3=paragraph_question)
    ifL(transitive, ('x', 'y','z'), orL( andL(is_more, 'x', is_more, 'y',is_more, 'z'), andL(is_more, 'x', is_less, 'y',is_less, 'z')))



print("Sensor part:")

paragraph_question['paragraph'] = ReaderSensor(keyword='paragraph')
paragraph_question['question'] = ReaderSensor(keyword='question')
paragraph_question['ques_id'] = ReaderSensor(keyword='ques_id')

paragraph_question[is_more] = ReaderSensor(keyword='more')
paragraph_question[is_less] = ReaderSensor(keyword='less')
paragraph_question[no_effect] = ReaderSensor(keyword='no_effect')

arg1, arg2 = pair.relate_to(word)
pair[s_arg1.reversed, s_arg2.reversed] = JointSensor(paragraph_question['text'], forward=make_pair)
pair['emb'] = FunctionalSensor(arg1.reversed('emb'), arg2.reversed('emb'), forward=concat)
pair[work_for] = ModuleLearner('emb', module=Classifier(200))

pair[work_for] = FunctionalReaderSensor(pair[arg1.reversed], pair[arg2.reversed], keyword='wf', forward=pair_label, label=True)



class QACrossEntropy(nn.CrossEntropyLoss):
    def forward(self, input, target, *args, **kwargs):
        x=input
        ys,ye=target[:,0],target[:,1]
        return super().forward(x[:,:,0], ys, *args, **kwargs)+super().forward(x[:,:,1], ye, *args, **kwargs)

class Metric_daiurs(torch.nn.Module):
    def forward(self, input, target, weight=None, dim=None):
        start_logits  = torch.argmax(input[:,:,0],dim=1)
        end_logits = torch.argmax(input[:,:,1],dim=1)
        accuracy_s = np.sum(start_logits.detach().cpu().numpy()== target[:,0].detach().cpu().numpy())
        accuracy_e = np.sum(end_logits.detach().cpu().numpy()== target[:,1].detach().cpu().numpy())
        return {'accuracy_s_sum': accuracy_s, 'accuracy_e_sum': accuracy_e,"questions_num":target.shape[0]}

class PRF1Tracker_darius(MetricTracker):
    def __init__(self):
        super().__init__(Metric_daiurs())

    def forward(self, values):
        start=0
        end=0
        sample_num=0
        for i in values:
            start+=i["accuracy_s_sum"]
            end+=i["accuracy_e_sum"]
            sample_num+=i["questions_num"]
        return {'accuracy_s': start/sample_num, 'accuracy_e': end/sample_num}

program = POIProgram(graph, loss=MacroAverageTracker(QACrossEntropy()),metric=PRF1Tracker_darius())

import logging

device = 'auto'
logging.basicConfig(level=logging.INFO)

program.train(reader, train_epoch_num=1, Optim=torch.optim.Adam, device=device)
print('Training result:')
print(program.model.loss)
print(program.model.metric)


for node in program.populate(reader, device=device):
    print("context:")
    print(node.getAttribute('context'),"\n")
    for word_node in node.getChildDataNodes():
        print(word_node.getAttribute('text'))

