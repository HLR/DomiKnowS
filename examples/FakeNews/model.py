from cgitb import text
from modulefinder import Module
import sys

sys.path.append("../")
sys.path.append("../../")
from graph import graph
from regr.sensor.pytorch.sensors import FunctionalSensor, ReaderSensor, JointSensor
from regr.sensor.pytorch.learners import ModuleLearner
from regr.program import SolverPOIProgram, IMLProgram
from regr.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric
from regr.program.loss import BCEWithLogitsIMLoss, NBCrossEntropyLoss, NBCrossEntropyIMLoss, BCEWithLogitsLoss
from regr.program.model.pytorch import SolverModel, IMLModel
from regr.program.primaldualprogram import PrimalDualProgram
from regr.sensor.pytorch.sensors import ReaderSensor
from regr.sensor.pytorch.learners import ModuleLearner
from regr.sensor.pytorch.sensors import JointSensor
import torch
from torch import nn
from torch.nn import Linear, Dropout, ReLU
import transformers
from transformers import RobertaModel
from transformers import RobertaTokenizer
from dataset import load_annodata

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


def tokenize_text(text):
    t = tokenizer(text, padding='max_length', max_length=192,
                  truncation=True, return_tensors="pt")
    return t["input_ids"], t["attention_mask"]


# def tokenize_parent_texts(parent_texts):
#     t=[tokenizer(text, padding='max_length', max_length=192,
#                       truncation=True, return_tensors="pt")
#             for text in parent_texts]
#     inputs = []
#     masks = []
#     for i in t:
#         inputs.append(i["input_ids"])
#         masks.append(i["attention_mask"])
#     # print("-"*20)
#     # print(torch.stack(inputs, dim=0).squeeze(dim=0).shape[0])
#     print(torch.ones(torch.stack(inputs, dim=0).squeeze(dim=0).shape[0], 1).to(torch.int64).shape)
#     print(torch.stack(inputs, dim=0).squeeze(dim=0).to(torch.int64).shape)
#     print(torch.stack(masks, dim=0).squeeze(dim=0).to(torch.int64).shape)
#
#     # print("-"*20)
#     return torch.ones(torch.stack(inputs, dim=0).squeeze(dim=0).shape[0], 1).to(torch.int64), \
#            torch.stack(inputs, dim=0).squeeze(dim=0).to(torch.int64), \
#            torch.stack(masks, dim=0).squeeze(dim=0).to(torch.int64)

def binary_reader(label):
    # print(label)
    return label


def parent_reader(labels):
    # print(labels)
    return labels


def parent_labels(labels):
    return torch.ones(len(labels), 1), labels


class RobertaClassifier(torch.nn.Module):

    def __init__(self, num_outputs, dropout=0.5):
        super(RobertaClassifier, self).__init__()

        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(768, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.roberta(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear1(dropout_output)
        # print("----------------------")
        # print(linear_output)
        # print("----------------------")
        return linear_output


def model_declaration(device):
    graph.detach()

    # text_sequence = graph["TextSequence"]
    # category = graph["Category"]
    # parent_tags = graph["ParentTag"]
    #
    # text_sequence["Text"] = ReaderSensor(keyword="Text", device=device)
    # text_sequence["BinaryLabel"] = ReaderSensor(keyword="Label", device=device)
    # text_sequence["ParentTexts"] = ReaderSensor(keyword="Parent Text", device=device)
    # text_sequence["ParentLabels"] = ReaderSensor(keyword="Parent Labels", device=device)
    #
    # text_sequence["TokenText","mask"] = JointSensor("Text", forward=tokenize_text)
    # category[TextContains, "TokenParentTexts","ParentMasks"] = JointSensor(text_sequence["ParentTexts"], forward=tokenize_parent_texts)
    # category[TextContains, "TokenParentLabels"] = JointSensor(text_sequence["ParentLabels"], forward=parent_labels)
    #
    # text_sequence[category] = FunctionalSensor("BinaryLabel", forward=binary_reader, label=True)
    # category[parent_tags] = FunctionalSensor("TokenParentLabels", forward=parent_reader, label=True)
    #
    # text_sequence[category] = ModuleLearner("TokenText","mask", module=RobertaClassifier(num_outputs=2))
    # category[parent_tags] = ModuleLearner("TokenParentTexts","ParentMasks", module=RobertaClassifier(num_outputs=13))

    text_sequence = graph["TextSequence"]
    category = graph["Category"]
    parent_category = graph["ParentCategory"]
    sub_category = graph["SubCategory"]

    text_sequence["Text"] = ReaderSensor(keyword="Text", device=device)
    text_sequence["BinaryLabel"] = ReaderSensor(keyword="Label", device=device)
    text_sequence["TokenText", "mask"] = JointSensor("Text", forward=tokenize_text)
    text_sequence[category] = FunctionalSensor("BinaryLabel", forward=binary_reader, label=True)
    text_sequence[category] = ModuleLearner("TokenText", "mask", module=RobertaClassifier(num_outputs=2))

    text_sequence["ParentLabels"] = ReaderSensor(keyword="Parent Labels", device=device)
    text_sequence["ParentTokenText", "ParentMask"] = JointSensor("Text", forward=tokenize_text)
    text_sequence[parent_category] = FunctionalSensor("ParentLabels", forward=parent_reader, label=True)
    text_sequence[parent_category] = ModuleLearner("ParentTokenText", "ParentMask",
                                                   module=RobertaClassifier(num_outputs=13))

    text_sequence["SubLabels"] = ReaderSensor(keyword="Sub Labels", device=device)
    text_sequence["SubTokenText", "SubMask"] = JointSensor("Text", forward=tokenize_text)
    text_sequence[sub_category] = FunctionalSensor("SubLabels", forward=parent_reader, label=True)
    text_sequence[sub_category] = ModuleLearner("SubTokenText", "SubMask", module=RobertaClassifier(num_outputs=31))

    program = SolverPOIProgram(graph, inferTypes=['local/argmax'], loss=MacroAverageTracker(NBCrossEntropyLoss()),
                               metric={'softmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})
    return program
