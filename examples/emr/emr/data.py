from collections import OrderedDict, Counter
from itertools import chain

import torch
from torch.utils.data import DataLoader
from spacy.lang.en import English


from .conll import Conll04CorpusReader
from .data_spacy import reprocess


def build_vocab():
    pass
def save_vocab():
    pass
def load_vocab():
    pass

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
        import pdb; pdb.set_trace()
        tokens, labels, relations = zip(*samples)
        vocab_token_counter = Counter(map(lambda t: t.text, chain(*tokens)))
        vocab_token_counter = vocab_token_counter.most_common(max_vocab)
        vocab_list, _ = zip(*filter(lambda kv: kv[1] > least_count, vocab_token_counter))
        vocab_list = list(vocab_list)
        vocab_list.insert(0, '_PAD_')
        vocab_list.append('_UNK_')
        vocab = OrderedDict((token, idx) for idx, token in enumerate(vocab_list))
        return vocab

    def collate_fn(self, batch):
        entity_types = ['']
        relation_types = ['']
        batch_size = len(batch)
        max_len = max(*(len(tokens) for tokens, _, _ in batch), 5)  # make sure len >= 5
        tokens_tensor = torch.empty((batch_size, max_len), dtype=torch.long)
        tokens_tensor.fill_(self.vocab['_PAD_'])
        for batch_idx, (tokens, labels, relations) in enumerate(batch):
            for token_idx, token in enumerate(tokens):
                tokens_tensor[batch_idx, token_idx] = self.vocab[token.text] if token.text in self.vocab else self.vocab['_UNK_']
        #import pdb; pdb.set_trace()
        return tokens_tensor

    def __init__(self, path, reader=Conll04CorpusReader(), first=True, vocab=None, least_count=0, max_vocab=None, **kwargs):
        self.reader = reader
        self.path = path
        sentences_list, relations_list = self.reader(path)
        def process(sample):
            sentence, relations = sample
            tokens, labels, relations = reprocess(sentence, relations, first=first)
            tokens, labels, relations = self.process_one(tokens, labels, relations)
            return tokens, labels, relations
        samples = list(map(process, zip(sentences_list, relations_list)))
        self.vocab = vocab or self.build_vocab(samples, least_count=0, max_vocab=None)
        super().__init__(samples, collate_fn=self.collate_fn, **kwargs)
