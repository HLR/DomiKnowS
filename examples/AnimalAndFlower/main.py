import sys
import torch
from torch.utils.data import random_split, DataLoader
import argparse
from dataset import load_animals_and_flowers
from model import model_declaration

sys.path.append('.')
sys.path.append('../..')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--solver', help='the model solver', default='iml')
args = parser.parse_args()


def main():
    import logging
    logging.basicConfig(level=logging.INFO)
    trainset = load_animals_and_flowers()
    train_ds, val_ds = random_split(trainset, [4021, 500])
    solver = args.solver if args.solver is not None else 'iml'
    program = model_declaration(solver=solver)
    program.train(train_ds, valid_set=val_ds, train_epoch_num=10, Optim=lambda param: torch.optim.Adam(param, lr=.0001),
                  device=device)


if __name__ == '__main__':
    main()
