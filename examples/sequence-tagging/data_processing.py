### Chen Zheng 05/19/2022

# Reading the data in the CoNLL-2003 format
# The following function reads a file represented in the CoNLL-2003 format. In this format, each row corresponds to one token. 
# or each token, there is a word, a part-of-speech tag, a "shallow syntax" label, and the BIO-coded named entity label, separated by whitespace. 
# The sentences are separated by empty lines. Here is an example of a sentence.

# United NNP B-NP B-ORG
# Nations NNP I-NP I-ORG
# official NN I-NP O
# Ekeus NNP B-NP B-PER
# heads VBZ B-VP O
# for IN B-PP O
# Baghdad NNP B-NP B-LOC
# . . O
# The function reads the file in this format and returns a torchtext Dataset, which in turn consists of a number of Example. 
# We will use just the words and the BIO labels, for the input and output respectively.

import torch
from torch import nn
import time
import torchtext
import numpy as np

import random

from collections import defaultdict, Counter


def read_data(corpus_file, datafields):
    with open(corpus_file, encoding='utf-8') as f:
        examples = []
        words = []
        labels = []
        for line in f:
            line = line.strip()
            if not line:
                examples.append(torchtext.legacy.data.Example.fromlist([words, labels], datafields))
                words = []
                labels = []
            else:
                columns = line.split()
                words.append(columns[0])
                labels.append(columns[-1])
        print(len(examples))
        return torchtext.legacy.data.Dataset(examples[0:len(examples)//10*1], datafields)



# Evaluating the predicted named entities
# To evaluate our named entity recognizers, we compare the named entities predicted by the system to the entities in the gold standard. 
# We follow standard practice and compute precision and recall scores, as well as the harmonic mean of the precision and recall, known as the F-score.

# Please note that the precision and recall scores are computed with respect to the full named entity spans and labels. 
# To be counted as a correct prediction, the system needs to predict all words in the named entity correctly, 
# and assign the right type of entity label. We don't give any credits to partially correct predictions.


# Convert a list of BIO labels, coded as integers, into spans identified by a beginning, an end, and a label.
# To allow easy comparison later, we store them in a dictionary indexed by the start position.
def to_spans(l_ids, voc):
    spans = {}
    current_lbl = None
    current_start = None
    for i, l_id in enumerate(l_ids):
        l = voc[l_id]

        if l[0] == 'B': 
            # Beginning of a named entity: B-something.
            if current_lbl:
                # If we're working on an entity, close it.
                spans[current_start] = (current_lbl, i)
            # Create a new entity that starts here.
            current_lbl = l[2:]
            current_start = i
        elif l[0] == 'I':
            # Continuation of an entity: I-something.
            if current_lbl:
                # If we have an open entity, but its label does not
                # correspond to the predicted I-tag, then we close
                # the open entity and create a new one.
                if current_lbl != l[2:]:
                    spans[current_start] = (current_lbl, i)
                    current_lbl = l[2:]
                    current_start = i
            else:
                # If we don't have an open entity but predict an I tag,
                # we create a new entity starting here even though we're
                # not following the format strictly.
                current_lbl = l[2:]
                current_start = i
        else:
            # Outside: O.
            if current_lbl:
                # If we have an open entity, we close it.
                spans[current_start] = (current_lbl, i)
                current_lbl = None
                current_start = None
    return spans

# Compares two sets of spans and records the results for future aggregation.
def compare(gold, pred, stats):
    for start, (lbl, end) in gold.items():
        stats['total']['gold'] += 1
        stats[lbl]['gold'] += 1
    for start, (lbl, end) in pred.items():
        stats['total']['pred'] += 1
        stats[lbl]['pred'] += 1
    for start, (glbl, gend) in gold.items():
        if start in pred:
            plbl, pend = pred[start]
            if glbl == plbl and gend == pend:
                stats['total']['corr'] += 1
                stats[glbl]['corr'] += 1

# This function combines the auxiliary functions we defined above.
def evaluate_iob(predicted, gold, label_field, stats):
    # The gold-standard labels are assumed to be an integer tensor of shape
    # (max_len, n_sentences), as returned by torchtext.
    gold_cpu = gold.t().cpu().numpy()
    gold_cpu = list(gold_cpu.reshape(-1))

    # The predicted labels assume the format produced by pytorch-crf, so we
    # assume that they have been converted into a list already.
    # We just flatten the list.
    pred_cpu = [l for sen in predicted for l in sen]
    
    # Compute spans for the gold standard and prediction.
    gold_spans = to_spans(gold_cpu, label_field.vocab.itos)
    pred_spans = to_spans(pred_cpu, label_field.vocab.itos)
    # print(gold_spans)
    # print(pred_spans)
    # import sys
    # sys.exit()

    # Finally, update the counts for correct, predicted and gold-standard spans.
    compare(gold_spans, pred_spans, stats)

# Computes precision, recall and F-score, given a dictionary that contains
# the counts of correct, predicted and gold-standard items.
def prf(stats):
    if stats['pred'] == 0:
        return 0, 0, 0
    p = stats['corr']/stats['pred']
    r = stats['corr']/stats['gold']
    if p > 0 and r > 0:
        f = 2*p*r/(p+r)
    else:
        f = 0
    return p, r, f