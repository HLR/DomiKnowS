import torch
torch.manual_seed(0)

import random
random.seed(0)

import numpy as np
np.random.seed(0)

from tqdm import tqdm
import torch.utils.data
from torchtext.vocab import GloVe, vocab
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pickle

limit_spans = 40
limit_words = 20

glove_vectors = GloVe()

# with open('examples/srl/data/srl_data_train.pkl', 'rb') as file_in:
#     train_data = pickle.load(file_in)

with open('srl/data/srl_data_dev.pkl', 'rb') as file_in:
    dev_data = pickle.load(file_in)

def unpack_data(data, subset=1.0):
    sentence_raw = []
    sentences = []
    predicates = []
    args = []
    spans = []

    print('Unpacking')

    limit_idx = int(len(data) * subset)

    for i, dp in enumerate(tqdm(data, total=limit_idx)):
        if i >= limit_idx:
            break

        sentence_spans = []

        for sp in dp['spans'].keys():
            span_mask = torch.zeros(len(dp['sentence']))
            span_mask[sp[0]: sp[1] + 1] = 1
            sentence_spans.append(span_mask)

        if len(sentence_spans) > limit_spans:
            continue

        if len(dp['sentence']) > limit_words:
            continue

        sentence_raw.append(dp['sentence'])
        sentences.append(glove_vectors.get_vecs_by_tokens(dp['sentence']))
        predicates.append(dp['predicate'])
        args.append(dp['args'])
        spans.append(sentence_spans)

    lens = [len(s) for s in sentences]

    return {
        'sentences': sentences,
        'predicates': predicates,
        'args': args,
        'lens': lens,
        'sentence_raw': sentence_raw,
        'spans': spans
    }


class SRLDataset:
    def __init__(self, data_blob, num_spans, shuffle=False):
        self.data_blob = data_blob
        self.num_spans = num_spans
        self.shuffle = shuffle

        self.idx_mapping = list(range(len(self)))

        if self.shuffle:
            random.shuffle(self.idx_mapping)

    def __len__(self):
        return len(self.data_blob['sentences'])

    def __getitem__(self, iter_idx):
        idx = self.idx_mapping[iter_idx]

        result = {}

        # words
        result['words'] = torch.unsqueeze(self.data_blob['sentences'][idx], dim=0)

        spans_all = []

        # sort spans by length (shortest to longest)
        sorted_spans = sorted(self.data_blob['spans'][idx], key=lambda l: torch.sum(l))

        # spans
        for sp_idx in range(self.num_spans):
            if sp_idx < len(sorted_spans):
                result['span_%d' % sp_idx] = sorted_spans[sp_idx].unsqueeze(0)
                spans_all.append(sorted_spans[sp_idx])
            else:
                result['span_%d' % sp_idx] = torch.zeros((result['words'].shape[1],)).unsqueeze(0)
                spans_all.append(torch.zeros((result['words'].shape[1],)))

            '''if sp_idx < 1:
                result['span_%d' % sp_idx] = torch.tensor([1, 0, 0, 0]).unsqueeze(0)
                spans_all.append(sorted_spans[sp_idx])
            else:
                result['span_%d' % sp_idx] = torch.zeros(sorted_spans[0].shape).unsqueeze(0)
                spans_all.append(torch.zeros(sorted_spans[0].shape))'''

        result['spans_all'] = [spans_all]

        #result['span_0'] = torch.ones(self.data_blob['spans'][idx][0].shape)
        #result['span_1'] = torch.zeros(self.data_blob['spans'][idx][0].shape)

        # arg labels
        result['arg_label'] = torch.tensor([[x] for x in self.data_blob['args'][idx]])
        #print(any([torch.equal(result['arg_label'].squeeze(), s.long()) for s in result['spans_all'][0]]))

        # predicate
        result['predicate'] = torch.tensor(self.data_blob['predicates'][idx]).unsqueeze(-1).unsqueeze(0)

        if iter_idx == len(self) - 1 and self.shuffle:
            random.shuffle(self.idx_mapping)

        return result


def select_batch(data, i_start, batch_size):
    return {
        k: v[i_start: i_start + batch_size] for k, v in data.items()
    }


data_blob = unpack_data(dev_data, subset=0.1)

del dev_data
del glove_vectors

print('number of instances loaded:', len(data_blob['sentences']))

# data_blob_train = select_batch(data_blob, 0, 5000)
# data_blob_train_mini = select_batch(data_blob, 0, 500)
data_blob_valid = select_batch(data_blob, 0, 1000)

# train_dataset = SRLDataset(data_blob_train, limit_spans)
# train_mini_dataset = SRLDataset(data_blob_train_mini, limit_spans)
valid_dataset = SRLDataset(data_blob_valid, limit_spans)

# print(data_blob_train)
