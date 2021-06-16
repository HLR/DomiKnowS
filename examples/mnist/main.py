import logging
from functools import partial
import torch

import config
from model import model_declaration
from data import get_readers


def main():
    logging.basicConfig(level=logging.INFO)

    program = model_declaration(config)
    
    trainreader, testreader = get_readers()

    program.train(trainreader, test_set=testreader, train_epoch_num=config.epochs, Optim=partial(torch.optim.SGD, lr=0.01))


if __name__ == '__main__':
    main()
