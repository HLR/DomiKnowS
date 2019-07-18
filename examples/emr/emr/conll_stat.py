import logging
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from tqdm import tqdm
from .conll import Conll04CorpusReader


class Conll04Stat():
    logger = logging.getLogger(__name__)
    read = Conll04CorpusReader()

    def __init__(self):
        pass

    def __call__(self, path):
        sentence_length = []
        for sentence, relations in tqdm(zip(*self.read(path))):
            (tokens, pos_tags, labels) = sentence
            # rel, (src, src_val), (dst, dst_val) = relation for relation in relations
            sentence_length.append(len(tokens))
        sentence_length = np.array(sentence_length)
        print('Sentence length min:{}, max:{}, mean:{}, mid:{}'.format(
            sentence_length.min(),
            sentence_length.max(),
            sentence_length.mean(),
            np.median(sentence_length),
        ))
        count, length = np.histogram(sentence_length)
        cum_count = count.cumsum()
        sentence_df = pd.DataFrame(data=[length, count, cum_count],
                                   index=['length', 'count', 'cum_count'],
                                   columns=length)
        print(sentence_df)


parser = ArgumentParser(
    description='Show stat of the data set.')
parser.add_argument('path', nargs=1)
args = parser.parse_args()


def main():
    stat = Conll04Stat()
    stat(args.path[0])


if __name__ == '__main__':
    main()
