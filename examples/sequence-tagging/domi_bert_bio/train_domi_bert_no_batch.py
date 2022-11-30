### Chen Zheng 05/19/2022

import sys
sys.path.append(".")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

import torch
from torch import nn
import numpy as np

from regr.sensor.pytorch.sensors import ReaderSensor, ConcatSensor, FunctionalSensor, JointSensor, JointReaderSensor
from regr.sensor.pytorch.learners import ModuleLearner
import graph

from regr.program import POIProgram, IMLProgram, SolverPOIProgram
from regr.program.metric import MacroAverageTracker, PRF1Tracker, PRF1Tracker, DatanodeCMMetric
from regr.program.loss import NBCrossEntropyLoss

from model_domi import BIO_Model

import torch.nn.functional as F
# from seqeval.metrics import accuracy_score, f1_score, classification_report
from torch.utils import data
from tqdm import trange, tqdm
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup

from data_set import BIOProcessor, BIODataSet

from regr.program.primaldualprogram import PrimalDualProgram
from regr.program.lossprogram import SampleLossProgram
from regr.program.model.pytorch import SolverModel

device = "cuda:6"
# device = "cpu"

######################################################################
# Data Reader
######################################################################
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_epochs = 5
batch_size=1
# data_dir='domi_bert_bio/data'
data_dir='data'
max_len=32

tokenizer = BertTokenizer.from_pretrained('./local_model_directory')
# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


bio_tagging_processor = BIOProcessor()

train_examples = bio_tagging_processor.get_train_examples(data_dir)
# train_examples = bio_tagging_processor.get_test_examples(data_dir)
val_examples = bio_tagging_processor.get_dev_examples(data_dir)
# test_examples = bio_tagging_processor.get_test_examples(data_dir)

tags_vals = bio_tagging_processor.get_labels()
label_map = {}

for (i, label) in enumerate(tags_vals):
    if '-' in label:
        label = '_'.join(label.split('-'))
    label_map[label] = i

# print(label_map) ### {'O': 0, 'B_MISC': 1, 'I_MISC': 2, 'B_PER': 3, 'I_PER': 4, 'B_ORG': 5, 'I_ORG': 6, 'B_LOC': 7, 'I_LOC': 8, '[CLS]': 9, '[SEP]': 10, 'X': 11}
# {'O': 0, 'B_MISC': 1, 'I_MISC': 2, 'B_PER': 3, 'I_PER': 4, 'B_ORG': 5, 'I_ORG': 6, 'B_LOC': 7, 'I_LOC': 8, 'X': 9}
# print(label_map.keys())

train_dataset = BIODataSet(data_list=train_examples[:9000], tokenizer=tokenizer, label_map=label_map, max_len=max_len)
eval_dataset = BIODataSet(data_list=val_examples, tokenizer=tokenizer, label_map=label_map, max_len=max_len)
# test_dataset = BIODataSet(data_list=test_examples, tokenizer=tokenizer, label_map=label_map, max_len=max_len)

train_iter = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
eval_iter = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
# test_iter = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

######################################################################
# Data Processing
######################################################################

def generate_data(data_iteration):
    data = []
    for batch in data_iteration:
    # for batch in test_iter:
        b_input_ids, b_labels, b_input_mask, b_token_type_ids, b_label_masks = batch
        # one_hot = F.one_hot(b_labels, num_classes=len(label_map))
        data.append({
            'text': b_input_ids.view(1, max_len),
            'input_mask': b_input_mask.view(1, max_len),
            'token_type_ids': b_token_type_ids.view(1, max_len),
            'labels': b_labels.view(max_len),
            'label_masks': b_label_masks.view(1, max_len),
        })
    return data

train_examples = generate_data(train_iter)
valid_examples = generate_data(eval_iter)
# test_examples = generate_data(test_iter)


######################################################################
# Graph Declaration
######################################################################
# from graph import graph, sentence, word, b_loc, i_loc, b_per, i_per, b_org, i_org, b_misc, i_misc, o, pad, bos
from graph import graph, sentence, word, labels, sen_word_rel
graph.detach()


# def forward_tensor(x):
#     words = []
#     rels = []
#     total = 0
#     for sentence in x:
#         words.extend(sentence)
#         rels.append((total, total + len(sentence)))
#         total += len(sentence)

#     connection = torch.zeros(total, len(x))
#     for sid, rel in enumerate(rels):
#         connection[rel[0]: rel[1]][:] = 1

#     words = torch.LongTensor(words)
#     return connection, words

def forward_tensor(x,x2,x3,x4):
    words = []
    input_mask = []
    token_type_ids = []
    label_masks = []

    rels = []
    total = 0
    for i in range(len(x)):
        words.extend(x[i])
        input_mask.extend(x2[i])
        token_type_ids.extend(x3[i])
        label_masks.extend(x4[i])
        rels.append((total, total + len(x[i])))
        total += len(x[i])

    connection = torch.zeros(total, len(x))
    for sid, rel in enumerate(rels):
        connection[rel[0]: rel[1]][:] = 1

    words = torch.LongTensor(words)
    input_mask = torch.LongTensor(input_mask)
    token_type_ids = torch.LongTensor(token_type_ids)
    label_masks = torch.LongTensor(label_masks)
    return connection, words, input_mask, token_type_ids, label_masks


print('start the ReaderSensor!')

sentence['text'] = ReaderSensor(keyword='text', device=device)
sentence['input_mask'] = ReaderSensor(keyword='input_mask')
sentence['token_type_ids'] = ReaderSensor(keyword='token_type_ids')
sentence['label_masks'] = ReaderSensor(keyword='label_masks')

# word[sen_word_rel[0], 'text', 'input_mask', 'token_type_ids', 'label_masks'] = JointSensor(sentence['text'], sentence['input_mask'], sentence['token_type_ids'], sentence['label_masks'], forward=forward_tensor)
word[sen_word_rel[0], 'text', 'input_mask', 'token_type_ids', 'label_masks'] = JointSensor(sentence['text'], sentence['input_mask'], sentence['token_type_ids'], sentence['label_masks'], forward=forward_tensor, device=device)

word[labels] = ReaderSensor(keyword='labels',label=True, device=device)
print('start the ModuleLearner!')

model = BIO_Model.from_pretrained('./local_model_directory', num_labels=len(label_map)).to(device)
# model = BIO_Model.from_pretrained('bert-base-cased', num_labels=len(label_map)).to(device)
word[labels] = ModuleLearner('text', 'input_mask', 'token_type_ids', 'label_masks', module=model, device=device)

######################################################################
# ILP OR PD OR SAMPLELOSS
######################################################################

##### ILP
program = SolverPOIProgram(graph, inferTypes=['ILP', 'local/argmax'], poi=(sentence, word),
                        loss=MacroAverageTracker(NBCrossEntropyLoss()),
                        metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
                                'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})

##### PD
# program = PrimalDualProgram(graph, SolverModel, poi=(sentence, word),inferTypes=['local/argmax'],loss=MacroAverageTracker(NBCrossEntropyLoss()),beta=1.0)

##### SampleLoss
# program = SampleLossProgram(
#     graph, SolverModel,
#     poi=(sentence, word),
#     inferTypes=['local/argmax'],
#     sample = True,
#     sampleSize=2,
#     sampleGlobalLoss = True,
#     metric={'sampleloss': PRF1Tracker(DatanodeCMMetric()),
#             'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))}
#     )

print('finish Graph Declaration')

######################################################################
# Train the model
######################################################################

program.train(train_examples, train_epoch_num=num_epochs, Optim=lambda param: torch.optim.Adam(param, lr=0.01, weight_decay=1e-5), device=device)

program.save("domi_ilp_5_bert")
# program.save("domi_pd_10")
# program.save("domi_sample_loss_10")

######################################################################
# Evaluate the model
######################################################################

program.load("domi_ilp_5_bert")
# program.load("domi_pd_10")
# program.load("domi_sample_loss_10")


from regr.utils import setProductionLogMode

productionMode = False
    

import logging
logging.basicConfig(level=logging.INFO)
program.test(valid_examples, device=device)




