### Chen Zheng 05/19/2022

import sys
sys.path.append(".")
sys.path.append("../")
sys.path.append("../../")

import torch
from torch import nn
import time
import torchtext
import numpy as np

import concept
from regr.sensor.pytorch.sensors import ReaderSensor, ConcatSensor, FunctionalSensor, JointSensor
from regr.sensor.pytorch.learners import ModuleLearner

# from torchtext.legacy import data

import random

from collections import defaultdict, Counter


from data_processing import read_data, evaluate_iob, prf
from model import RNNCRFTagger

import matplotlib.pyplot as plt

# %config InlineBackend.figure_format = 'retina' 
# plt.style.use('seaborn')

class Tagger:
    
    def __init__(self, lower):
        self.TEXT = torchtext.legacy.data.Field(init_token='<bos>', eos_token='<eos>', sequential=True, lower=lower)
        self.LABEL = torchtext.legacy.data.Field(init_token='<bos>', eos_token='<eos>', sequential=True, unk_token=None)
        self.fields = [('text', self.TEXT), ('label', self.LABEL)]
        # self.device = "cuda:0"
        self.device = "cpu"
        
    def tag(self, sentences):
        # This method applies the trained model to a list of sentences.
        
        # First, create a torchtext Dataset containing the sentences to tag.
        examples = []
        for sen in sentences:
            labels = ['?']*len(sen) # placeholder
            examples.append(torchtext.legacy.data.Example.fromlist([sen, labels], self.fields))
        dataset = torchtext.legacy.data.Dataset(examples, self.fields)
        
        iterator = torchtext.data.Iterator(
            dataset,
            device=self.device,
            batch_size=64,
            repeat=False,
            train=False,
            sort=False)
        
        # Apply the trained model to all batches.
        out = []
        self.model.eval()
        with torch.no_grad():
            for batch in iterator:
                # Call the model's predict method. This returns a list of NumPy matrix
                # containing the integer-encoded tags for each sentence.
                predicted = self.model.predict(batch.text)

                # Convert the integer-encoded tags to tag strings.
                for tokens, pred_sen in zip(sentences, predicted):
                    out.append([self.LABEL.vocab.itos[pred_id] for _, pred_id in zip(tokens, pred_sen[1:])])
        return out

    def program_declaration(self, text_emb, label_emb):
        ### domiknows start
        # from concept import graph, sentence, word, pad, bos, eos, out_bio, i_PER, i_ORG, i_LOC, i_MISC, b_MISC, b_ORG, b_LOC, sent_word_contains, label, lable_list
        from concept import graph, sentence, word, sent_word_contains, label, label_list
        graph.detach()
        ### domiknows end

        ### text tokens and labels
        sentence['tokenized_text'] = ReaderSensor(keyword='sentecne')
        sentence['tokenized_label'] = ReaderSensor(keyword='label')
        word[sent_word_contains, "token", 'label'] = JointSensor(sentence['tokenized_text'],sentence['tokenized_label'])

        word[label] = ModuleLearner("predict_model", module=RNNCRFTagger(text_emb, label_emb, emb_dim=300, rnn_size=128, update_pretrained=False))
        ### read sudoku

        from regr.program import POIProgram, IMLProgram, SolverPOIProgram
        from regr.program.metric import MacroAverageTracker, PRF1Tracker, PRF1Tracker, DatanodeCMMetric
        from regr.program.loss import NBCrossEntropyLoss

        # Creating the program to create model
        program = SolverPOIProgram(graph, inferTypes=['ILP', 'local/argmax'],
                                loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
                                        'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})

        return program   


    def train(self):

        # Read training and validation data according to the predefined split.
        # train_examples = read_data('data/eng.train.iob', self.fields)
        # valid_examples = read_data('data/eng.valid.iob', self.fields)
        train_examples = read_data('data/eng.train', self.fields)
        valid_examples = read_data('data/eng.testa', self.fields)

        # Count the number of words and sentences.
        n_tokens_train = 0
        n_sentences_train = 0
        for ex in train_examples:
            n_tokens_train += len(ex.text) + 2
            n_sentences_train += 1
        n_tokens_valid = 0       
        for ex in valid_examples:
            n_tokens_valid += len(ex.text)

        # Load the pre-trained embeddings that come with the torchtext library.
        use_pretrained = True
        if use_pretrained:
            print('We are using pre-trained word embeddings.')
            self.TEXT.build_vocab(train_examples, vectors="/data/hlr/glove840B/glove.840B.300d")
        else:  
            print('We are training word embeddings from scratch.')
            self.TEXT.build_vocab(train_examples, max_size=5000)
        self.LABEL.build_vocab(train_examples)


        ### use domi code begin
        program = self.program_declaration(self.TEXT, self.LABEL)
        ### use domi code end

    
        # Create one of the models defined above.
        #self.model = RNNTagger(self.TEXT, self.LABEL, emb_dim=300, rnn_size=128, update_pretrained=False)
        # self.model = RNNCRFTagger(self.TEXT, self.LABEL, emb_dim=300, rnn_size=128, update_pretrained=False)
    
        # self.model.to(self.device)
    
        batch_size = 1024
        n_batches = np.ceil(n_sentences_train / batch_size)

        mean_n_tokens = n_tokens_train / n_batches

        train_iterator = torchtext.legacy.data.BucketIterator(
            train_examples,
            device=self.device,
            batch_size=batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            train=True,
            sort=True)

        valid_iterator = torchtext.legacy.data.BucketIterator(
            valid_examples,
            device=self.device,
            batch_size=64,
            sort_key=lambda x: len(x.text),
            repeat=False,
            train=False,
            sort=True)
    
        train_batches = list(train_iterator)
        valid_batches = list(valid_iterator)

        # optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=1e-5)

        n_labels = len(self.LABEL.vocab)
        # print(self.LABEL.vocab.itos)
        # # ['<pad>', '<bos>', '<eos>', 'O', 'I-PER', 'I-ORG', 'I-LOC', 'I-MISC', 'B-MISC', 'B-ORG', 'B-LOC']
        # print(n_labels) ### 11

        history = defaultdict(list)    
        
        n_epochs = 25

        for i in range(1, n_epochs + 1):
            program.train(train_batches, train_epoch_num=1, Optim=lambda param: torch.optim.Adam(param, lr = 0.01, weight_decay=1e-5 ), device='cpu')
        
#         for i in range(1, n_epochs + 1):

#             t0 = time.time()

#             loss_sum = 0

#             self.model.train()
#             for batch in train_batches:
                
#                 # # Compute the output and loss.
#                 # loss = self.model(batch.text, batch.label) / mean_n_tokens
                
#                 # optimizer.zero_grad()            
#                 # loss.backward()
#                 # optimizer.step()
#                 # loss_sum += loss.item()
#                 program.train(batch.text, Optim=lambda param: SGD(param, lr=args.learning_rate),device='cpu')

#             train_loss = loss_sum / n_batches
#             history['train_loss'].append(train_loss)

#             # Evaluate on the validation set.
#             if i % 1 == 0:
#                 stats = defaultdict(Counter)

#                 self.model.eval()
#                 with torch.no_grad():
#                     for batch in valid_batches:
#                         # Predict the model's output on a batch.
#                         predicted = self.model.predict(batch.text)                   
#                         # Update the evaluation statistics.
#                         evaluate_iob(predicted, batch.label, self.LABEL, stats)
            
#                 # Compute the overall F-score for the validation set.
#                 _, _, val_f1 = prf(stats['total'])
                
#                 history['val_f1'].append(val_f1)
            
#                 t1 = time.time()
#                 print(f'Epoch {i}: train loss = {train_loss:.4f}, val f1: {val_f1:.4f}, time = {t1-t0:.4f}')
           
#         # After the final evaluation, we print more detailed evaluation statistics, including
#         # precision, recall, and F-scores for the different types of named entities.
#         print()
#         print('Final evaluation on the validation set:')
#         p, r, f1 = prf(stats['total'])
#         print(f'Overall: P = {p:.4f}, R = {r:.4f}, F1 = {f1:.4f}')
#         for label in stats:
#             if label != 'total':
#                 p, r, f1 = prf(stats[label])
#                 print(f'{label:4s}: P = {p:.4f}, R = {r:.4f}, F1 = {f1:.4f}')
        
#         plt.plot(history['train_loss'])
#         plt.plot(history['val_f1'])
#         plt.legend(['training loss', 'validation F-score'])

tagger = Tagger(lower=False)
tagger.train()


# def print_tags(sentence):
#     tokens = sentence.split()
#     tags = tagger.tag([tokens])[0]
#     for token, tag in zip(tokens, tags):
#         print(f'{token:12s}{tag}')


# print_tags('John Johnson was born in Moscow , lives in Gothenburg , and works for Chalmers Technical University and the University of Gothenburg .')

# print_tags('Paris Hilton lives in New York .')

# print_tags('New York Stock Exchange is in New York .')