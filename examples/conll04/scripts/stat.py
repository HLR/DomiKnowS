import logging
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

from utils import add_example_path

add_example_path()

from conll.data.reader import Conll04CorpusReader

class Stat():
    def setup(self): pass
    def sample(self, sample): pass
    def summarize(self): pass


class Length(Stat):
    def setup(self):
        self.sentence_length = []
    def sample(self, sample):
        sentence, relations = sample
        (tokens, pos_tags, labels) = sentence
        self.sentence_length.append(len(tokens))
    def summarize(self):
        sentence_length = np.array(self.sentence_length)
        yield 'Sentence length / min', sentence_length.min()
        yield 'Sentence length / max', sentence_length.max()
        yield 'Sentence length / mean', sentence_length.mean()
        yield 'Sentence length / mid', np.median(sentence_length)
        count, length = np.histogram(sentence_length)
        cum_count = count.cumsum()
        sentence_df = pd.DataFrame(data=[length, count, cum_count],
                                   index=['length', 'count', 'cum_count'],
                                   columns=length)
        yield 'Sentence length / hist\n', sentence_df


class Labels(Stat):
    def setup(self):
        self.labels = Counter()
    def sample(self, sample):
        sentence, relations = sample
        (tokens, pos_tags, labels) = sentence
        self.labels.update(labels)
    def summarize(self):
        for label, count in self.labels.most_common():
            yield 'Labels count / {}'.format(label), count

# [r'NN.?', 'CD', 'JJ']
def pairArgCandidate(tag):
    return 'NN' in tag or 'CD' in tag or 'JJ' in tag
missed_entity = []
missed_pair_arg = []

import itertools
class EntitiesWithPosTag(Stat):
    NON_ENEITY_LABELS = ['O']
    def setup(self):
        self.pos_tags = Counter()
    def sample(self, sample):
        sentence, relations = sample
        (tokens, pos_tags, labels) = sentence
        filtered = list(filter(lambda x: x[1][1] not in self.NON_ENEITY_LABELS, enumerate(zip(pos_tags, labels))))
        if filtered:
            idxs, filtered_pos_tags = zip(*filtered)
            filtered_pos_tags, _ = zip(*filtered_pos_tags)
            filtered_pos_tags = list(filtered_pos_tags)
            # VB, FW/FW, PRP, VB/PRP, CC, VBN
            # if 'FW/FW' in filtered_pos_tags:
            #     print(tokens, '\n', pos_tags, '\n', labels)
            missed_pos_tags = list(filter(lambda x: not pairArgCandidate(x[1]), zip(idxs, filtered_pos_tags)))
            for missed_pos_tag in missed_pos_tags:
                missed_entity.append((tokens, pos_tags, missed_pos_tag))
                print(f'\nMissed Entity {len(missed_entity)}\n{(tokens, pos_tags, missed_pos_tag)}')
                for (rel_type, (arg1_idx, _), (arg2_idx, _)), (idx, pos_tag) in itertools.product(relations, missed_pos_tags):
                    if idx in {arg1_idx, arg2_idx}:
                        missed_pair_arg.append((tokens, pos_tags, (rel_type, arg1_idx, arg2_idx, idx)))
                        print(f'\nMissed Relation {len(missed_pair_arg)}\n{(rel_type, arg1_idx, arg2_idx)}')
            self.pos_tags.update(filtered_pos_tags)
    def summarize(self):
        for pos_tag, count in self.pos_tags.most_common():
            yield 'Entity Pos-tag count / {}'.format(pos_tag), count

class Relations(Stat):
    def setup(self):
        self.counter = Counter()
    def sample(self, sample):
        sentence, relations = sample
        self.counter.update([rel for rel, _, _ in relations])
    def summarize(self):
        for label, count in self.counter.most_common():
            yield 'Relation count / {}'.format(label), count


class Conll04Stat():
    logger = logging.getLogger(__name__)
    read = Conll04CorpusReader()

    def __init__(self):
        self.Stats = [Length(), Labels(), EntitiesWithPosTag(), Relations()]

    def __call__(self, path):
        for stat in self.Stats:
            stat.setup()
        for sentence, relations in tqdm(zip(*self.read(path))):
            # (tokens, pos_tags, labels) = sentence
            # rel, (src, src_val), (dst, dst_val) = relation for relation in relations
            for stat in self.Stats:
                stat.sample((sentence, relations))
                
        for stat in self.Stats:
            for name, result in stat.summarize():
                self.logger.info('- {}: {}'.format(name, result))
                yield name, result


def main():
    parser = ArgumentParser(
        description='Show stat of the data set.')
    parser.add_argument('path', nargs=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    stat = Conll04Stat()
    path = args.path[0]
    for name, result in stat(path):
        pass


if __name__ == '__main__':
    main()
