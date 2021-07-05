import sys
import torch
from torch.utils.data import random_split

sys.path.append('.')
sys.path.append('../..')

from dataset import load_animals_and_flowers

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from model import model_declaration, train_with_single_network


def main():
    import logging
    logging.basicConfig(level=logging.INFO)
    trainset = load_animals_and_flowers()
    train_ds, val_ds = random_split(trainset, [4021, 500])
    program = model_declaration()
    program.train(train_ds, valid_set=val_ds, train_epoch_num=10, Optim=lambda param: torch.optim.Adam(param, lr=.0001),
                  device=device)

    # train_with_single_network(train_ds,val_ds,['monkey', 'cat', 'squirrel', 'dog', 'daisy', 'dandelion', 'rose' ,'tulip', 'sunflower','flower', 'animal'])


if __name__ == '__main__':
    main()
