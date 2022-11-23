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

# device = "cuda:0"
device = "cpu"

######################################################################
# Data Reader
######################################################################
# train_sentences = load_examples('../bio_data/train.txt', True) ### True: Replace every digit in a string by a zero.
train_sentences = load_examples('../bio_data/testb.txt', True)
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
    # print('----------------->:', connection.shape, len(words), words)
    return connection, words

print('start the ReaderSensor!')

sentence['text'] = ReaderSensor(keyword='text')

# word[b_loc] = ReaderSensor(keyword='b_loc',label=True, device=device)
# word[i_loc] = ReaderSensor(keyword='i_loc',label=True, device=device)
# word[b_per] = ReaderSensor(keyword='b_per',label=True, device=device)
# word[i_per] = ReaderSensor(keyword='i_per',label=True, device=device)
# word[b_org] = ReaderSensor(keyword='b_org',label=True, device=device)
# word[i_org] = ReaderSensor(keyword='i_org',label=True, device=device)
# word[b_misc] = ReaderSensor(keyword='b_misc',label=True, device=device)
# word[i_misc] = ReaderSensor(keyword='i_misc',label=True, device=device)
# word[o] = ReaderSensor(keyword='o',label=True, device=device)
# word[pad] = ReaderSensor(keyword='pad',label=True, device=device)
# word[bos] = ReaderSensor(keyword='bos',label=True, device=device)
word[labels] = ReaderSensor(keyword='labels',label=True, device=device)

word[sen_word_rel[0], 'text'] = JointSensor(sentence['text'], forward=forward_tensor) ## what is the meaning
# word[sen_word_rel[0], 'words', 'labels'] = JointSensor(sentence['words'], word[labels], forward=forward_tensor) ## what is the meaning

# tmp = torch.nn.Parameter(glove.vectors, requires_grad=False) ### torch.Size([2196017, 300])

print('start the ModuleLearner!')
# word[emb] = ModuleLearner('emb', module=RNNTagger(glove, tag_vocab, emb_dim=300, rnn_size=128, update_pretrained=False), device=device)
# word[b_loc] = ModuleLearner('emb', module=RNNTagger(glove, tag_vocab, emb_dim=300, rnn_size=128, update_pretrained=False), device=device)
# word[i_loc] = ModuleLearner('emb', module=RNNTagger(glove, tag_vocab, emb_dim=300, rnn_size=128, update_pretrained=False), device=device)
# word[b_per] = ModuleLearner('emb', module=RNNTagger(glove, tag_vocab, emb_dim=300, rnn_size=128, update_pretrained=False), device=device)
# word[i_per] = ModuleLearner('emb', module=RNNTagger(glove, tag_vocab, emb_dim=300, rnn_size=128, update_pretrained=False), device=device)
# word[b_org] = ModuleLearner('emb', module=RNNTagger(glove, tag_vocab, emb_dim=300, rnn_size=128, update_pretrained=False), device=device)
# word[i_org] = ModuleLearner('emb', module=RNNTagger(glove, tag_vocab, emb_dim=300, rnn_size=128, update_pretrained=False), device=device)
# word[b_misc] = ModuleLearner('emb', module=RNNTagger(glove, tag_vocab, emb_dim=300, rnn_size=128, update_pretrained=False), device=device)
# word[i_misc] = ModuleLearner('emb', module=RNNTagger(glove, tag_vocab, emb_dim=300, rnn_size=128, update_pretrained=False), device=device)
# word[o] = ModuleLearner('emb', module=RNNTagger(glove, tag_vocab, emb_dim=300, rnn_size=128, update_pretrained=False), device=device)
# word[pad] = ModuleLearner('emb', module=RNNTagger(glove, tag_vocab, emb_dim=300, rnn_size=128, update_pretrained=False), device=device)
# word[bos] = ModuleLearner('emb', module=RNNTagger(glove, tag_vocab, emb_dim=300, rnn_size=128, update_pretrained=False), device=device)
word[labels] = ModuleLearner('text', module=RNNTagger(glove, tag_vocab, emb_dim=300, rnn_size=128, update_pretrained=False), device=device)

### why the above ModuleLearners are so slow

# Creating the program to create model
program = SolverPOIProgram(graph, inferTypes=['ILP', 'local/argmax'],
                        loss=MacroAverageTracker(NBCrossEntropyLoss()),
                        metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
                                'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})

print('finish Graph Declaration')

######################################################################
# Train the model
######################################################################
n_epochs = 1
batch_size = 1024
n_batches = np.ceil(len(train_examples) / batch_size)

print(train_examples[0])
program.train(train_examples, train_epoch_num=n_epochs, Optim=lambda param: torch.optim.Adam(param, lr=0.01, weight_decay=1e-5), device=device)