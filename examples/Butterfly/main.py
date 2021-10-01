import sys

sys.path.append('/home/admiraldarius/DomiKnowS/')
sys.path.append('/home/admiraldarius/DomiKnowS/regr')
sys.path.append('/home/hlr/storage/egr/research-hlr/nafarali/new_meta/DomiKnowS/regr')
sys.path.append('/home/hlr/storage/egr/research-hlr/nafarali/new_meta/DomiKnowS')

import argparse
from dataset import loadDataset
from model import model_declaration
import logging, random
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', dest='cuda_number', default=0, help='cuda number to train the models on', type=int)
parser.add_argument('--solver', help='the model solver', default='primal_dual')
parser.add_argument('--samplenum', dest='samplenum', default=4521, help='number of samples to choose from the dataset',
                    type=int)
parser.add_argument('--verbose', dest='verbose', default=0, help='print the errors', type=int)
parser.add_argument('--debugmode', dest='debugmode', default=0, help='different debugging setting to test the code',
                    type=int)
parser.add_argument('--epochs', dest='epochs', default=30, help='number of training epoch', type=int)
parser.add_argument('--lambdaValue', dest='lambdaValue', default=0.1, help='value of learning rate', type=float)
parser.add_argument('--model_size', dest='model_size', default=10, help='size of the model', type=int)
parser.add_argument('--lr', dest='learning_rate', default=2e-5, help='learning rate of the adam optimiser', type=float)

# TODO add  model size and other things here
args = parser.parse_args()

import math


def main():
    if torch.cuda.is_available():
        device = "cuda:" + str(args.cuda_number)
    else:
        device = 'cpu'
    print("selected device is:", device)

    torch.manual_seed(777)
    random.seed(777)
    logging.basicConfig(level=logging.INFO)
    # trainset = loadDataset(mode="train")
    train_ds = loadDataset(mode="test")
    import gc
    gc.collect()
    print("Data Loaded")
    solver = args.solver
    program = model_declaration(device=device, solver=solver, lambdaValue=args.lambdaValue, model_size=args.model_size)
    program.train(train_ds, train_epoch_num=args.epochs,
                  Optim=lambda param: torch.optim.Adam(param, lr=args.learning_rate), device=device)


if __name__ == '__main__':
    main()
