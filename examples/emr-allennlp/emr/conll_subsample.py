import logging
import random
from argparse import ArgumentParser
from .conll_split import Conll04Processor


class Conll04Subsample(Conll04Processor):
    logger = logging.getLogger(__name__)

    def __call__(self, path, portion, subfix='.corp'):
        examples = self.read(path)
        total_num = len(examples)
        sub_num = int(portion * total_num)
        sub_examples = random.sample(examples, sub_num)
        self.write('{}_subsample_{}{}'.format(path, portion, subfix), sub_examples)
        return sub_examples


def main():
    parser = ArgumentParser(
        description='Subsample Conll04 dataset for experiments with less data.')
    parser.add_argument('path', nargs=1)
    parser.add_argument('-p', '--portion', nargs='?', type=float, default=0.5,
                        help='Portion of data to keep in the new dataset. Default: 0.5.')
    parser.add_argument('-S', '--subfix', nargs='?', default='.corp',
                        help='subfix for the output file name. Default: ".corp".')
    args = parser.parse_args()

    processor = Conll04Subsample()
    sub_examples = processor(args.path[0], args.portion, args.subfix)
    print('New dataset with {} examples is created.'.format(len(sub_examples)))


if __name__ == '__main__':
    main()
