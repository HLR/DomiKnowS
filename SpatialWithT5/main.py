import argparse

from program_declaration import program_declaration
from reader import DomiKnowS_reader
import torch
import numpy as np
import random


def train(program, train_set, epoch, lr, cur_device):
    optimizer = lambda param: torch.optim.AdamW(param, lr=lr)
    program.train(train_set, c_warmup_iters=0, train_epoch_num=epoch,
                  Optim=optimizer,
                  device=cur_device)


def main(args):
    SEED = 382
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    cuda_number = args.cuda
    if cuda_number == -1:
        cur_device = 'cpu'
    else:
        cur_device = "cuda:" + str(cuda_number) if torch.cuda.is_available() else 'cpu'

    train_data = DomiKnowS_reader("DataSet/train_FR_v3.json", "FR",
                                  size=args.train_size,
                                  upward_level=12,
                                  augmented=True,
                                  batch_size=args.batch_size,
                                  STEPGAME_status=None)

    test_data = DomiKnowS_reader("DataSet/test.json", "FR",
                                 size=args.test_size,
                                 augmented=False,
                                 batch_size=args.batch_size,
                                 STEPGAME_status=None,
                                 )

    dev_data = DomiKnowS_reader("DataSet/dev.json", "FR",
                                size=args.test_size,
                                augmented=False,
                                batch_size=args.batch_size,
                                STEPGAME_status=None,
                                )

    program = program_declaration(device=cur_device, pmd=args.pmd, beta=args.beta)

    train(program, train_data, epoch=args.epoch, lr=args.lr, cur_device=cur_device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run SpaRTUN Rules Base")
    parser.add_argument("--epoch", dest="epoch", type=int, default=1)
    parser.add_argument("--lr", dest="lr", type=float, default=1e-5)
    parser.add_argument("--cuda", dest="cuda", type=int, default=0)
    parser.add_argument("--test_size", dest="test_size", type=int, default=12)
    parser.add_argument("--train_size", dest="train_size", type=int, default=16)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=4)
    parser.add_argument("--pmd", dest="pmd", type=bool, default=False)
    parser.add_argument("--beta", dest="beta", type=float, default=0.5)
    args = parser.parse_args()
    main(args)
