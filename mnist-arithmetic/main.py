import logging
from functools import partial
import torch

import config
from model import model_declaration
from data import get_readers

from domiknows.utils import setProductionLogMode

def validate(program, reader):
    from graph import digit, summation
    for node in program.populate(reader, device='auto'):
        #print(node)
        # node.infer()
        node.inferILPResults()
        image1, image2 = node.relationLinks['contains']
        addition, = image1.impactLinks['operand1']
        assert addition is image2.impactLinks['operand2'][0]

        x = image1.getAttribute(digit, 'ILP').argmax()
        y = image2.getAttribute(digit, 'ILP').argmax()
        z = addition.getAttribute(summation, 'ILP').argmax()
        print(f'{x}+{y}={z}')
        if x + y != z:
            print("False\n")
            
        verifyResult = node.verifyResultsLC()
        if verifyResult:
            for lc in verifyResult:
                print("lc %s is %i%% satisfied by learned results"%(lc, verifyResult[lc]['satisfied']))

def main():
   
    setProductionLogMode()

    #logging.basicConfig(level=logging.ERROR)

    program = model_declaration(config)
    
    trainreader, testreader = get_readers()

    # validate graph
    validate(program, trainreader)

    # do training
    program.train(trainreader, test_set=testreader, train_epoch_num=config.epochs, Optim=partial(torch.optim.SGD, lr=config.lr), device='auto')


if __name__ == '__main__':
    main()
