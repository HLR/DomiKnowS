import os
import logging
import random
import math
import itertools
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm


class Conll04Processor():
    logger = logging.getLogger(__name__)

    def read(self, path):
        with open(path, 'r') as fin:
            lines = [line for line in tqdm(fin)]

        examples = []
        example = ''
        blank = 0
        for line in tqdm(lines):
            if not examples and not example and not line.strip():
                # skip blanks in the begining or in the end
                continue
            if not line.strip():
                blank += 1
            example += line
            if blank == 2:
                # add example
                if example:
                    examples.append(example)
                # reset
                example = ''
                blank = 0
        if example:
            examples.append(example)

        return examples

    def write(self, path, examples):
        with open(path, 'w') as fout:
            for example in tqdm(examples):
                fout.write(example)

class Conll04Splitter(Conll04Processor):
    def create_shuffle_shuffle(self, n):
        # `random.shuffle` has a limited random period 2**19937-1 that
        # implies at most 2080 possible shuffle will be generated and
        # picked from
        elems = list(range(n))
        random.shuffle(elems)
        return elems

    def create_shuffle_sample(self, n):
        elems = range(n)
        elems = random.sample(elems, k=n)
        return elems

    def create_split_random(self, total_num, splits_num):
        index_list = self.create_shuffle_sample(total_num)
        step = int(math.floor(total_num / splits_num))
        reminder = total_num - step * splits_num
        offset = 0
        for split_index in range(reminder):
            yield index_list[offset:(offset + step + 1)]
            offset += step + 1
        for split_index in range(reminder, splits_num):
            yield index_list[offset:(offset + step)]
            offset += step
        assert offset == total_num

    def sample_by_splits(self, examples, splits):
        for split in splits:
            yield [examples[index] for index in split]

    def __call__(self, path, splits_num, subfix='.corp', valid=False):
        examples = self.read(path)
        total_num = len(examples)
        splits = self.create_split_random(total_num, splits_num)
        example_splits = list(self.sample_by_splits(examples, splits))
        for split_index in tqdm(reversed(range(splits_num)), total=splits_num):
            test_index = split_index
            valid_index = (split_index + 1) % splits_num if valid else -1
            train_split = itertools.chain.from_iterable(
                (example_splits[j] for j in range(splits_num) if j not in [test_index, valid_index]))
            self.write('{}_{}_train{}'.format(
                path, split_index + 1, subfix), train_split)
            test_split = example_splits[test_index]
            self.write('{}_{}_test{}'.format(
                path, split_index + 1, subfix), test_split)
            if not valid:
                continue
            valid_split = example_splits[valid_index]
            self.write('{}_{}_valid{}'.format(
                path, split_index + 1, subfix), valid_split)
        return example_splits

def main():
    parser = ArgumentParser(
        description='Split Conll04 dataset for cross-validation.')
    parser.add_argument('path', nargs=1)
    parser.add_argument('-s', '--splits-num', nargs='?', type=int, default=5,
                        help='number of folds the dataset will be splitted equally into. Default: 5.')
    parser.add_argument('-v', '--valid', action='store_true', default=False,
                        help='create validation set with one of the folds.')
    parser.add_argument('-S', '--subfix', nargs='?', default='.corp',
                        help='subfix for the output file name. Default: ".corp".')
    args = parser.parse_args()

    spliter = Conll04Splitter()
    example_splits = spliter(args.path[0], args.splits_num, args.subfix, valid=args.valid)
    print('{}-fold datasets with {} examples are created.'.format(len(example_splits), tuple(len(split) for split in example_splits)))


if __name__ == '__main__':
    main()
