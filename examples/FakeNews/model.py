from modulefinder import Module
import sys
sys.path.append("../")
sys.path.append("../../")
from graph import graph
from regr.sensor.pytorch.sensors import FunctionalSensor, ReaderSensor
from regr.sensor.pytorch.learners import ModuleLearner
from regr.program import SolverPOIProgram, IMLProgram
from regr.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric
from regr.program.loss import BCEWithLogitsIMLoss, NBCrossEntropyLoss, NBCrossEntropyIMLoss
from regr.program.model.pytorch import SolverModel, IMLModel
from regr.program.primaldualprogram import PrimalDualProgram
from regr.sensor.pytorch.sensors import ReaderSensor
from regr.sensor.pytorch.learners import ModuleLearner
import torch
from torch import nn
from torch.nn import Linear, Dropout, ReLU
import transformers
from transformers import RobertaModel
from transformers import RobertaTokenizer
from dataset import load_annodata

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def tokenize_text(text):
    return tokenizer(text, padding='max_length', max_length=192, 
                     truncation=True, return_tensors="pt") 

def tokenize_parent_texts(parent_texts):
    return [tokenizer(text, padding='max_length', max_length=192, 
                      truncation=True, return_tensors="pt") 
            for text in parent_texts]

def binary_reader(label):
    return label

def parent_reader(labels):
    print("-"*40)
    print(labels)
    print("-"*40)
    return labels

class RobertaClassifier(torch.nn.Module):

    def __init__(self, num_outputs, dropout=0.5):

        super(RobertaClassifier, self).__init__()

        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(768, num_outputs)
        self.relu = nn.ReLU()
      

    def forward(self, input_id, mask):
        _, pooled_output = self.roberta(input_ids=input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear1(dropout_output)
        return linear_output

def model_declaration(device):
    graph.detach()
    
    text_sequence = graph["TextSequence"]
    category = graph["Category"]
    parent_tags = graph["ParentTag"]

    text_sequence["Text"] = ReaderSensor(keyword="Text", device=device)
    text_sequence["BinaryLabel"] = ReaderSensor(keyword="Label", device=device)
    text_sequence["ParentTexts"] = ReaderSensor(keyword="Parent Text", device=device)
    text_sequence["ParentLabels"] = ReaderSensor(keyword="Parent Labels", device=device)
    
    text_sequence["TokenText"] = FunctionalSensor("Text", forward=tokenize_text)
    #category["TokenParentTexts"] = FunctionalSensor("Parent Text", forward=tokenize_parent_texts)

    text_sequence["BinaryLabel"] = FunctionalSensor("BinaryLabel", forward=binary_reader, label=True)
    #category["ParentLabels"] = FunctionalSensor("ParentLabels", forward=parent_reader, label=True)

    text_sequence[category] = ModuleLearner("TokenText", module=RobertaClassifier(num_outputs=1))

    program = SolverPOIProgram(graph,poi=[text_sequence,text_sequence[category]], loss=MacroAverageTracker(NBCrossEntropyLoss()),
                               metric=PRF1Tracker(DatanodeCMMetric('local/argmax')))
    return program

