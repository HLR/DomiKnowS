import logging
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from .conll import Conll04CorpusReader

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
        self.Stats = [Length(), Labels(), Relations()]

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
    for name, result in stat(args.path[0]):
        pass


if __name__ == '__main__':
    main()
