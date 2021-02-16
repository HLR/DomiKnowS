
import wget
import os
import torch
import numpy as np
from regr.program import POIProgram

from torch import nn
from regr.program.metric import MacroAverageTracker, PRF1Tracker, MetricTracker, CMWithLogitsMetric
from regr.program.loss import NBCrossEntropyLoss

from regr.sensor.pytorch.learners import ModuleLearner
from regr.sensor.pytorch.sensors import ReaderSensor, TorchEdgeSensor, JointSensor, FunctionalSensor, \
    FunctionalReaderSensor

print('Downloading dataset')

url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
url_2 = "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt"

if not os.path.exists('data/squad1.1/'):
    os.makedirs("data/squad1.1")

if not os.path.exists('data/squad1.1/train-v1.1.json'):
    wget.download(url, 'data/squad1.1/train-v1.1.json')

if not os.path.exists('data/squad1.1/bert-base-uncased-vocab.txt'):
    wget.download(url_2, 'data/squad1.1/bert-base-uncased-vocab.txt')

from read_QA import QA_reader_2, DariusQABERT_2,make_questions,QA_Tokenize,PutNear
max_lenght=256
out = QA_reader_2(sample_num=10)  # out = Context,Questions,Answer_start,Answer_end

from regr.graph import Graph, Concept, Relation

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('QA_graph') as graph:
    paragraph = Concept(name='paragraph')
    question = Concept(name='question')
    para_quest_rel, = paragraph.contains(question)

reader = []
for context,questions,answer_start,answer_end in zip(*out):
    reader.append({"context": [context], "questions": [questions], "answer_start": [answer_start], "answer_end": [answer_end]})
print("Reader Formed:")
print("reader size:",len(reader))
print(reader[0]["context"])
print(reader[0]["questions"])
print(reader[0]["answer_start"])
print(reader[0]["answer_end"],"\n")

print("Sensor part:")
paragraph['context'] = ReaderSensor(keyword='context')
paragraph['questions'] = ReaderSensor(keyword='questions')
paragraph['answer_start'] = ReaderSensor(keyword='answer_start')
paragraph['answer_end'] = ReaderSensor(keyword='answer_end')

context = paragraph.relate_to(question)[0]
question[context,"context2", 'text',"start","end"] = JointSensor(paragraph['context'],paragraph['questions'],paragraph['answer_start'],paragraph['answer_end'], forward=make_questions)

question["token_ids","Mask","Sent_number","offsets"] = JointSensor("context2", 'text', forward=QA_Tokenize(max_lenght=max_lenght))

question['label'] = FunctionalSensor('start', 'end',"token_ids","offsets", forward=PutNear(max_lenght),label=True)

question["label"] = ModuleLearner("token_ids", "Mask", "Sent_number", module=DariusQABERT_2())


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


