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

device = "cuda:1"
# device = "cpu"

######################################################################
# Data Reader
######################################################################
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_epochs = 5
batch_size=1
data_dir='data'
max_len=32

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


bio_tagging_processor = BIOProcessor()

train_examples = bio_tagging_processor.get_train_examples(data_dir)
val_examples = bio_tagging_processor.get_dev_examples(data_dir)
test_examples = bio_tagging_processor.get_test_examples(data_dir)

tags_vals = bio_tagging_processor.get_labels()
label_map = {}

for (i, label) in enumerate(tags_vals):
    if '-' in label:
        label = '_'.join(label.split('-'))
    label_map[label] = i

print(label_map) ### {'O': 0, 'B_MISC': 1, 'I_MISC': 2, 'B_PER': 3, 'I_PER': 4, 'B_ORG': 5, 'I_ORG': 6, 'B_LOC': 7, 'I_LOC': 8, '[CLS]': 9, '[SEP]': 10, 'X': 11}

train_dataset = BIODataSet(data_list=train_examples, tokenizer=tokenizer, label_map=label_map, max_len=max_len)
eval_dataset = BIODataSet(data_list=val_examples, tokenizer=tokenizer, label_map=label_map, max_len=max_len)
test_dataset = BIODataSet(data_list=test_examples, tokenizer=tokenizer, label_map=label_map, max_len=max_len)

# print(len(train_dataset[:10000]))

train_iter = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
eval_iter = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_iter = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

######################################################################
# Data Processing
######################################################################

def generate_data(data_iteration):
    data = []
    # for batch in data_iteration:
    for batch in test_iter:
        b_input_ids, b_labels, b_input_mask, b_token_type_ids, b_label_masks = batch
        # print('-------------->>>>>', b_input_ids, b_input_ids.size()) ### torch.Size([32, 128])
        # print('-------------->>>>>', b_input_mask, b_input_mask.size()) ### torch.Size([32, 128])
        # print('-------------->>>>>', b_token_type_ids, b_token_type_ids.size()) ### torch.Size([32, 128])
        # print('-------------->>>>>', b_labels, b_labels.size()) ### torch.Size([32, 128])
        # print('-------------->>>>>', b_label_masks, b_label_masks.size()) ### torch.Size([32, 128])
        # one_hot = F.one_hot(b_labels, num_classes=len(label_map))
        data.append({
            'text': b_input_ids.view(1, max_len),
            'input_mask': b_input_mask.view(max_len),
            'token_type_ids': b_token_type_ids.view(max_len),
            'labels': b_labels.view(max_len),
            # 'labels': one_hot.view(max_len),
            'label_masks': b_label_masks.view(max_len),
        })
    return data


train_examples = generate_data(train_iter)
valid_examples = generate_data(eval_iter)
test_examples = generate_data(test_iter)


######################################################################
# Graph Declaration
######################################################################
# from graph import graph, sentence, word, b_loc, i_loc, b_per, i_per, b_org, i_org, b_misc, i_misc, o, pad, bos
from graph import graph, sentence, word, labels, sen_word_rel
graph.detach()


######################################################################
# I have problem on the JointSensor forward_tensor again.
######################################################################

### x1 is 'input_id', x2 is 'input_mask', x3 is 'token_type_ids', x4 is 'label_masks' 
### the size of x1, x2, x3, x4 are all torch.Size([32, 128])


# def forward_tensor(input_id, input_mask, token_type_ids, label_masks):
#     ### how to write connection code?

#     ### return connection, input_id, input_mask, token_type_ids, label_masks

#     # connection = torch.zeros(len(x), total)
#     # for sid, rel in enumerate(rels):
#     #     connection[sid][rel[0]: rel[1]] = 1

#     # idx = input_mask.nonzero()[:, 0].unsqueeze(-1)
#     # connection = torch.zeros(idx.shape[0], idx.max()+1)
#     # connection.scatter_(1, idx, 1)
#     # connection = connection.view(connection.size(1), connection.size(0))


#     # print(connection)
#     # print(connection.size()) ## (batch_size, 807)

#     # connection = torch.ones(input_id.size(0), 1, input_id.size(1))
#     connection = torch.ones(input_id.size(1), input_id.size(0))
#     # print(connection.size())
#     # import sys
#     # sys.exit()

#     return connection, input_id, input_mask, token_type_ids, label_masks


# def forward_tensor(x, input_mask, token_type_ids, label_masks):
def forward_tensor(x):
    words = []
    rels = []
    total = 0
    for sentence in x:
        words.extend(sentence)
        rels.append((total, total + len(sentence)))
        total += len(sentence)

    connection = torch.zeros(len(x), total)
    for sid, rel in enumerate(rels):
        connection[sid][rel[0]: rel[1]] = 1

    words = torch.LongTensor(words)
    # print('connection: ', connection.size()) ## torch.Size([1, 32])
    # print('words: ', words.size()) ## torch.Size([32])
    # return connection, words, input_mask, token_type_ids, label_masks
    return connection, words






print('start the ReaderSensor!')

sentence['text'] = ReaderSensor(keyword='text', device=device)
# sentence['input_mask'] = ReaderSensor(keyword='input_mask')
# sentence['token_type_ids'] = ReaderSensor(keyword='token_type_ids')
# sentence['label_masks'] = ReaderSensor(keyword='label_masks')

word[labels] = ReaderSensor(keyword='labels',label=True, device=device)

# word[sen_word_rel[0], 'text', 'input_mask', 'token_type_ids', 'label_masks'] = JointSensor(sentence['text'], sentence['input_mask'], sentence['token_type_ids'], sentence['label_masks'], forward=forward_tensor)
word[sen_word_rel[0], 'text'] = JointSensor(sentence['text'], forward=forward_tensor, device=device)
print('start the ModuleLearner!')

model = BIO_Model.from_pretrained('bert-base-cased', num_labels=len(label_map)).to(device)
word[labels] = ModuleLearner('text', module=model, device=device)

# Creating the program to create model
program = SolverPOIProgram(graph, inferTypes=['ILP', 'local/argmax'],
                        loss=MacroAverageTracker(NBCrossEntropyLoss()),
                        metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
                                'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})

print('finish Graph Declaration')

######################################################################
# Train the model
######################################################################

# program.train(train_examples, train_epoch_num=num_epochs, Optim=lambda param: torch.optim.Adam(param, lr=0.01, weight_decay=1e-5), device=device)

# program.save("domi_0")
# print('model saved!!!')

######################################################################
# Evaluate the model
######################################################################

program.load("domi_0") # in case we want to load the model instead of training


from regr.utils import setProductionLogMode

productionMode = False
    

import logging
logging.basicConfig(level=logging.INFO)
program.test(valid_examples, device=device)




