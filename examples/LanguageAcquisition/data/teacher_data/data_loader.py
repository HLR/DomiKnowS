'''
    This script contains all the support functions related to the datasets
'''

import sys

sys.path.append("../models")
sys.path.append("../dataset")

import time
import copy
import math

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker


# This is used to make sure that the program can use GPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data_logic(fp, n=None, islabel=False):
    '''
        Reads the filename and creates the list of the logical forms,
        input sentence, and target sentence.
        
        Parameters
            filename (file pointer object): Ascii file with the data separated by \t
            n (int): Number of sentences to load
            
        Returns
            sentence: List of word sequences
            logical_form: List of logical forms sequences
            ground_sentences: List of the ground truth word sequences
            ground_situations: List of ground truth predicates
    '''
    
    
    #data to return
    #sentence, logical_form, ground_sentence, label = [],[],[],[]
    data = []
    
    count = 0
    
    #Iterate through the file
    for line in fp:
        
        if n != None and count >= n:
            break
        
        
        line = line.split("\t")
        
        try:
            
            if islabel:
                label = line[-1].strip().split()
            
        
            logical_form = line[0].strip().split()
            ground_sentence = line[1].strip().split()
            #ground_sentence = ground_sentence[:11] if len(ground_sentence) >= 12 else ground_sentence
            
            tup = (logical_form, ground_sentence)
            data.append(tup)
        
                
        except IndexError:
            
            # if len(line) == 3:
            #     if label == ['0']:
            #         sentence = [' ']
            #     else:
            #         logical_form = [' ']
                    
            # else:
                print("Error:",line)
                
        # tup = (sentence, logical_form, ground_situation, ground_sentence)
        # data.append(tup)
        
        count += 1
        
    #return sentence,logical_form,ground_sentence,label
    return data


def build_data_tensors_logic(data, V, P):
    '''
        This function builds the tensor of each example in the data
        
        Parameters:
            data: List of tuples, where each tuple has the following
                sentence: input utterance
                predicates: input situation
                ground_sentence: ground truth utterance
                label: 0-1 (True or False) label
            V: List of vocabulary words in the data
            P: List of predicates in the data
            C: List of categories in the data
                
    '''
    
    data_lst = []
    for elem in data:
        #print(elem)
        
        #elem = elem.split("\t")
        
        sentence = elem[1]#.split()
        predicate = elem[0]#.split()
        
        #print(sentence)
        #print(predicate)
        #print(ground_sentence)
        sentence_tensor = create_input_tensor(sentence, V)
        predicate_tensor = create_logic_tensor(predicate, P)
        
        # combine each tensor as one
        
        tup = (predicate_tensor, sentence_tensor)
        
        #tup = torch.stack(tup, dim=0)
        #print(tup.shape)
        data_lst.append(tup)
    
    return data_lst



def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def build_list(filename):
    
    
    S = set()
    
    fp = open(filename,"r")
    
    for w in fp:
        w = w.strip()
        S.add(w)

    return sorted(S)


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()


def create_input_tensor(sentence,wordlist):
    '''
        Creates the tensor for the sentence.
        
        Parameters
            sentence: List of words in the sentence
            wordlist: List of all the words in the dataset
            
        Returns
            tensor: Tensor object of the sequence.
    '''
    
    indices = [wordlist.index(word) for word in sentence + ['<eos>']]
    
    tensor = torch.tensor(indices,dtype=torch.long, device=device)
    
    return tensor.view(-1,1)



def create_logic_tensor(logical_forms,logical_forms_lst):
    '''
        Creates the tensor of logical forms of the sequence.
        
        Parameters
            logical_forms: List of logical forms in the sequence
            logical_forms_lst: List of all the logical forms in the dataset
        
        Returns
            tensor: Tensor object of the sequence.
    '''
    
    #build a list of indices
    
    logicals = copy.deepcopy(logical_forms)
    logicals.append("<eos>")
    
    indices = [logical_forms_lst.index(logic) for logic in logicals]
    
    tensor = torch.tensor(indices,dtype=torch.long, device=device)
    
    return tensor.view(-1,1)


def create_target_tensor(sentence,wordlist):
    '''
        Creates the tensor of the target index of the words in the sequence.
        
        Parameters
            sentence: List of words in the target sentence
            wordlist: List of all the words in the dataset
            
        Returns
            tensor: Tensor object of the target sequence.
    '''
    
    indices = [wordlist.index(word) for word in sentence + ['<eos>']]
    
    tensor = torch.tensor(indices,dtype=torch.long, device=device)
    
    return tensor.view(-1,1)
    
def create_target_tensor_single(word, wordlist):
    '''
        Creates the tensor of the target index of the words in the sequence.
        
        Parameters
            sentence: List of words in the target sentence
            wordlist: List of all the words in the dataset
            
        Returns
            tensor: Tensor object of the target sequence.
    '''
    
    indices = [wordlist.index(word) if word != '<eos>' else wordlist.index('<eos>')]
    
    tensor = torch.tensor(indices,dtype=torch.long, device=device)
    
    return tensor.view(-1,1)
    