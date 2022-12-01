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

from model_domi import RNNTagger
from data_reader_no_torchtext import load_examples, word_mapping, char_mapping, tag_mapping, lower_case

from regr.program.primaldualprogram import PrimalDualProgram
from regr.program.lossprogram import SampleLossProgram
from regr.program.model.pytorch import SolverModel

device = "cuda:5"
# device = "cpu"

######################################################################
# Data Reader
######################################################################
# train_sentences = load_examples('bio_data/train.txt', True) ### True: Replace every digit in a string by a zero.
# dev_sentences = load_examples('bio_data/testa.txt', True)
# test_sentences = load_examples('bio_data/testb.txt', True)
train_sentences = load_examples('../bio_data/train.txt', True) ### True: Replace every digit in a string by a zero.
dev_sentences = load_examples('../bio_data/testa.txt', True)
test_sentences = load_examples('../bio_data/testb.txt', True)

### load glove vector and glove vocab
from torchtext.vocab import GloVe
glove = GloVe()
glove_vocab = glove.stoi

### tagging labels vocab and id
dico_tags, tag_vocab, id_to_tag = tag_mapping(train_sentences) ### tag_vocab: tag_to_id

######################################################################
# Data Processing
######################################################################

def generate_data(sentences, word_to_id, tag_to_id, lower=False):
    data = []
    for s in sentences:
        str_words = [w[0] for w in s]
        words = [word_to_id[lower_case(w,lower) if lower_case(w,lower) in word_to_id else 'unk']
                 for w in str_words]
        labels = [tag_to_id[w[-1]] for w in s]
        data.append({
            'fullsentencestr': str_words, ## string 
            # 'text': [words],
            # 'labels': labels, ## label
            'text': torch.LongTensor([words]),
            'labels': torch.LongTensor(labels), ## label
        })
    return data

train_examples = generate_data(train_sentences, glove_vocab, tag_vocab, lower=True)
valid_examples = generate_data(dev_sentences, glove_vocab, tag_vocab, lower=True)
test_examples = generate_data(test_sentences, glove_vocab, tag_vocab, lower=True)

print("{} / {} / {} sentences in train / dev / test.".format(len(train_examples), len(valid_examples), len(test_examples))) ## 14041 / 3250 / 3453 sentences in train / dev / test.14041 / 3250 / 3453 sentences in train / dev / test.

print(test_examples[0])


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
#         # print(words.extend(sentence))
#         # print(rels)
#         # print(total)
#         # print('<><><>'*50)

#     connection = torch.zeros(len(x), total)
#     for sid, rel in enumerate(rels):
#         connection[sid][rel[0]: rel[1]] = 1

#     words = torch.LongTensor(words)
#     return connection, words

def forward_tensor(x):
    words = []
    rels = []
    total = 0
    for sentence in x:
        words.extend(sentence)
        rels.append((total, total + len(sentence)))
        total += len(sentence)

    connection = torch.zeros(total, len(x))
    for sid, rel in enumerate(rels):
        connection[rel[0]: rel[1]][:] = 1

    words = torch.LongTensor(words)
    return connection, words

print('start the ReaderSensor!')

sentence['text'] = ReaderSensor(keyword='text', device=device)

word[sen_word_rel[0], 'text'] = JointSensor(sentence['text'], forward=forward_tensor, device=device) ## what is the meaning

word[labels] = ReaderSensor(keyword='labels',label=True, device=device)

print('start the ModuleLearner!')
word[labels] = ModuleLearner('text', module=RNNTagger(glove, tag_vocab, emb_dim=300, rnn_size=128, update_pretrained=False), device=device)


## Creating the program to create model
program = SolverPOIProgram(graph, inferTypes=['ILP', 'local/argmax'], poi=(sentence, word),
                        loss=MacroAverageTracker(NBCrossEntropyLoss()),
                        metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
                                'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})

# program = PrimalDualProgram(graph, SolverModel, poi=(sentence, word),inferTypes=['local/argmax'],loss=MacroAverageTracker(NBCrossEntropyLoss()),beta=1.0)


# program = SampleLossProgram(
#     graph, SolverModel,
#     poi=(sentence, word),
#     inferTypes=['local/argmax'],
#     sample = True,
#     sampleSize=2,
#     sampleGlobalLoss = True
#     )

print('finish Graph Declaration')

######################################################################
# save model
######################################################################

# def compute_scores(item, criteria="P"):
#         entities = ["location", "people", "organization", "other"]
#         instances = {"location": 937, "people": 774, "organization": 512, "other": 610, "work_for": 71, "located_in": 75, "live_in": 103, 
#                      "orgbase_on": 97, "kill": 55} ### ???
#         sum_entity = 0
#         sum_relations = 0
#         precision_entity = 0
#         precision_relations = 0
#         normal_precision_entity = 0
#         normal_precision_relations = 0
#         sum_all = 0
#         precision_all = 0
#         normal_precision_all = 0
#         for key in entities:
#             sum_entity += float(instances[key])
#             precision_entity += float(instances[key]) * float(item[key][criteria])
#             normal_precision_entity += float(item[key][criteria])

#         sum_all = sum_relations + sum_entity
#         precision_all = precision_entity + precision_relations
#         normal_precision_all = normal_precision_relations + normal_precision_entity

#         outputs = {}
        
#         if criteria == "P":
#             outputs["micro_" + str(criteria) + "_entities"] = precision_entity / sum_entity
#             outputs["micro_" + str(criteria) + "_relations"] = precision_relations / sum_relations
#             outputs["micro_" + str(criteria) + "_all"] = precision_all / sum_all

#         outputs["macro_" + str(criteria) + "_entities"] = normal_precision_entity / len(entities)
        
#         return outputs

# def save_best(program, epoch=1, best_epoch=-1, best_macro_f1=0):
#         import logging
#         logger = logging.getLogger(__name__)
#         metrics = program.model.metric['argmax'].value()
#         results = compute_scores(metrics, criteria="F1")
#         score = results["macro_F1_all"]
#         if score > best_macro_f1:
#             logger.info(f'New Best Score {score} achieved at Epoch {epoch}.')
#             best_epoch = epoch
#             best_macro_f1 = score
#             if args.number == 1:
#                 program.save(f'saves/conll04-bert-{split_id}-best-macro-f1.pt')
#             else:
#                 program.save(f'saves/conll04-bert-{split_id}-size-{args.number}-best_macro-f1.pt')
#         return epoch + 1, best_epoch, best_macro_f1

######################################################################
# Train the model
######################################################################
n_epochs = 15
# batch_size = 1024
# n_batches = np.ceil(len(train_examples) / batch_size)

program.train(train_examples, train_epoch_num=n_epochs, Optim=lambda param: torch.optim.Adam(param, lr=0.01, weight_decay=1e-5), device=device)

program.save("domi_ilp_epoch_20")
# program.save("domi_pd_epoch_1")
# program.save("domi_sampleloss_epoch_1")

######################################################################
# Evaluate the model
######################################################################

program.load("domi_ilp_epoch_20")
# program.load("domi_pd_epoch_1")
# program.load("domi_sampleloss_epoch_1")


from regr.utils import setProductionLogMode

productionMode = False
# if productionMode:
#     setProductionLogMode(no_UseTimeLog=False)
    

import logging
logging.basicConfig(level=logging.INFO)
program.test(valid_examples, device=device)
