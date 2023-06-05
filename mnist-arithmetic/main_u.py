import logging
from functools import partial
import torch

import config
from model_u import model_declaration
from data_u import get_readers

from domiknows.utils import setProductionLogMode

def validate(program, reader):
    from graph_u import digit, summationVal
    for node in program.populate(reader, device='auto'):
        #print(node)
        # node.infer()
        node.inferILPResults()
        image1, image2 = node.relationLinks['contains']
        addition, = image1.impactLinks['operand1']
        assert addition is image2.impactLinks['operand2'][0]

        x = image1.getAttribute(digit, 'ILP').argmax()
        y = image2.getAttribute(digit, 'ILP').argmax()
        print(f'{x}+{y}={summationVal}')
        if x + y != summationVal:
            print("False\n")

def main():

    setProductionLogMode()

    logging.basicConfig(level=logging.INFO)

    program = model_declaration(config)
    
    trainreader, testreader = get_readers()

    # validate graph
    validate(program, trainreader)

    # do training
    program.train(trainreader, test_set=testreader, train_epoch_num=config.epochs, Optim=partial(torch.optim.SGD, lr=config.lr), device='auto')


if __name__ == '__main__':
    main()
