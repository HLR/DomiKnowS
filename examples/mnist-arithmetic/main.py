import logging
from functools import partial
import torch

import config
from model import model_declaration
from data import get_readers


def validate(program, reader):
    from graph import digit, summation
    for node in program.populate(reader, device='auto'):
        print(node)
        # node.infer()
        node.inferILPResults()
        image1, image2 = node.relationLinks['contains']
        addition, = image1.impactLinks['operand1']
        assert addition is image2.impactLinks['operand2'][0]

        print(image1.getAttribute(digit, 'ILP'))
        print(image2.getAttribute(digit, 'ILP'))
        print(addition.getAttribute(summation, 'ILP'))


def main():
    logging.basicConfig(level=logging.INFO)

    program = model_declaration(config)
    
    trainreader, testreader = get_readers()

    # validate graph
    validate(program, trainreader)

    # do training
    program.train(trainreader, test_set=testreader, train_epoch_num=config.epochs, Optim=partial(torch.optim.SGD, lr=config.lr), device='auto')


if __name__ == '__main__':
    main()
