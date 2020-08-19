from collections import OrderedDict, Counter
from itertools import chain

import torch
from torch.utils.data import DataLoader
from spacy.lang.en import English

from .conll import Conll04CorpusReader
from .data_spacy import reprocess


class ConllDataLoader(DataLoader):
    nlp = English()

    @classmethod
    def split(cls, sentence):
        return cls.nlp.tokenizer(sentence)

    @classmethod
    def match(cls, index, tokens, spacy_tokens):
        char_index = 0
        char_length = 0
        for index_cur, token in enumerate(tokens):
            if index_cur < index[0]:
                pass
            elif index_cur < index[1]:
                char_length += len(token) + 1
            else:
                break
            char_index += len(token) + 1
        # - 1 char_length has one more space, then we do not need -1
        start_char_index = char_index - char_length

        matched_spacy_tokens = []
        for spacy_token in spacy_tokens:
            if spacy_token.idx < start_char_index:
                pass
            elif spacy_token.idx < start_char_index + char_length:
                assert spacy_token.idx + \
                    len(spacy_token) <= start_char_index + char_length
                matched_spacy_tokens.append(spacy_token.i)
            else:
                break
        return matched_spacy_tokens

    @classmethod
    def process_one(cls, tokens, labels, relations):
        sentence = ' '.join(tokens)
        split_tokens = cls.split(sentence)

        split_labels = []
        for label in labels:
            (label_type, index, token) = label
            token = cls.match(index, tokens, split_tokens)
            split_label = (label_type, token)
            split_labels.append(split_label)

        split_relations = []
        for relation in relations:
            (relation_type, (src_index, src_token), (dst_index, dst_token)) = relation
            src_token = cls.match(src_index, tokens, split_tokens)
            dst_token = cls.match(dst_index, tokens, split_tokens)
            split_relation = (relation_type, src_token, dst_token)
            split_relations.append(split_relation)

        return split_tokens, split_labels, split_relations

    @classmethod
    def build_vocab(cls, samples, least_count=0, max_vocab=None):
        tokens, labels, relations = zip(*samples)
        # token vocab
        vocab_token_counter = Counter(map(lambda t: t.text, chain(*tokens)))
        vocab_token_counter = vocab_token_counter.most_common(max_vocab)
        vocab_list, _ = zip(*filter(lambda kv: kv[1] > least_count, vocab_token_counter))
        vocab_list = list(vocab_list)
        vocab_list.insert(0, '_PAD_')
        vocab_list.append('_UNK_')
        vocab = OrderedDict((token, idx) for idx, token in enumerate(vocab_list))
        # token lable vocab
        labels, _ = zip(*chain(*filter(None, labels)))
        vocab_label_counter = Counter(labels)
        label_list = list(vocab_label_counter.keys())
        label_vocab = OrderedDict((label, idx) for idx, label in enumerate(label_list))
        # relation vocab
        relations, _, _ = zip(*chain(*filter(None, relations)))
        vocab_relations_counter = Counter(relations)
        relation_list = list(vocab_relations_counter.keys())
        relation_vocab = OrderedDict((label, idx) for idx, label in enumerate(relation_list))
        #import pdb; pdb.set_trace()
        return {'token': vocab,
                'label': label_vocab,
                'relation': relation_vocab}

    def _collate_fn(self, batch):
        token_vocab = self.vocab['token']
        label_vocab = self.vocab['label']
        relation_vocab = self.vocab['relation']
        arg_idx = 0 if self.first else -1
        batch_size = len(batch)
        max_len = max(*(len(tokens) for tokens, _, _ in batch), 5)  # make sure len >= 5
        # initialize tensors
        tokens_tensor = torch.empty((batch_size, max_len), dtype=torch.long)
        tokens_tensor.fill_(token_vocab['_PAD_'])
        label_tensors = {}
        for label in label_vocab:
            label_tensors[label] = torch.empty((batch_size, max_len), dtype=torch.bool)
            label_tensors[label].fill_(False)
        relation_tensors = {}
        for relation in relation_vocab:
            relation_tensors[relation] = torch.empty((batch_size, max_len, max_len), dtype=torch.bool)
            relation_tensors[relation].fill_(False)
        for batch_idx, (tokens, labels, relations) in enumerate(batch):
            for token_idx, token in enumerate(tokens):
                tokens_tensor[batch_idx, token_idx] = token_vocab[token.text] if token.text in token_vocab else token_vocab['_UNK_']
            for label, arg in labels:
                label_tensors[label][batch_idx, arg[arg_idx]] = True
            for relation, arg1, arg2 in relations:
                relation_tensors[relation][batch_idx, arg1[arg_idx], arg2[arg_idx]] = True
            #import pdb; pdb.set_trace()
        token_raw, label_raw, relation_raw = zip(*batch)
        data_item = {
            'token_raw': list(token_raw),
            'label_raw': list(label_raw),
            'relation_raw': list(relation_raw),
            'token': tokens_tensor}
        data_item.update(label_tensors)
        data_item.update(relation_tensors)
        #import pdb; pdb.set_trace()
        return data_item

    def __init__(self, path, reader=Conll04CorpusReader(), first=True, vocab=None, least_count=0, max_vocab=None, skip_none=True, **kwargs):
        self.reader = reader
        self.path = path
        self.first = first
        sentences_list, relations_list = self.reader(path)
        def process(sample):
            sentence, relations = sample
            tokens, labels, relations = reprocess(sentence, relations, first=first, skip_none=skip_none)
            tokens, labels, relations = self.process_one(tokens, labels, relations)
            return tokens, labels, relations
        samples = list(map(process, zip(sentences_list, relations_list)))
        self.vocab = vocab or self.build_vocab(samples, least_count=least_count, max_vocab=max_vocab)
        super().__init__(samples, collate_fn=self._collate_fn, **kwargs)


def collate(batch):
    sentences, relations = zip(*batch)
    # (tokens, pos, label)
    # (relation_type, (src_index, src_token), (dst_index, dst_token))
    tokens, postags, labels = zip(*sentences)
    data_item = {
        'sentence': [' '.join(token_list) for token_list in tokens],
        'tokens': list(tokens),
        'postag': list(postags),
        'label': list(labels),
        'relation': list(relations),
    }
    #import pdb; pdb.set_trace()
    return data_item


class NaiveDataLoader(DataLoader):
    def __init__(self, path, reader=None, **kwargs):
        self.path = path
        self.reader = reader or Conll04CorpusReader()
        sentences_list, relations_list = self.reader(path)
        samples = list(zip(sentences_list, relations_list))
        super().__init__(samples, collate_fn=collate, **kwargs)
