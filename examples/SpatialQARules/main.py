import sys

sys.path.append(".")
sys.path.append("../")
sys.path.append("../../")

import pandas as pd
import torch
import argparse
import numpy as np
from regr.graph import Graph, Concept, Relation
from program_declaration import program_declaration
from reader import DomiKnowS_reader


def main(args):
    cuda_number = args.cuda
    if cuda_number == -1:
        cur_device = 'cpu'
    else:
        cur_device = "cuda:" + str(cuda_number) if torch.cuda.is_available() else 'cpu'

    program = program_declaration(cur_device,
                                  pmd=args.pmd, beta=args.beta,
                                  sampling=args.sampling, sampleSize=args.sampling_size,
                                  dropout=args.dropout, constrains=args.constrains)

    training_set = DomiKnowS_reader("DataSet/train_with_rules.json", "YN",
                                    size=args.train_size, upward_level=8, train=True, batch_size=args.batch_size)
    if args.loaded:
        program.load(args.loaded_file, map_location={'cuda:0': cur_device, 'cuda:1': cur_device})
    else:
        program.train(training_set, train_epoch_num=args.epoch,
                      Optim=lambda param: torch.optim.Adam(param, lr=args.lr, amsgrad=True),
                      device=cur_device)

    if args.save:
        program.save(args.save_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SpaRTUN Rules Base")
    parser.add_argument("--epoch", dest="epoch", type=int, default=1)
    parser.add_argument("--lr", dest="lr", type=float, default=1e-5)
    parser.add_argument("--cuda", dest="cuda", type=int, default=0)
    parser.add_argument("--test_size", dest="test_size", type=int, default=100000)
    parser.add_argument("--train_size", dest="train_size", type=int, default=100000)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=100000)
    parser.add_argument("--dropout", dest="dropout", type=bool, default=False)
    parser.add_argument("--pmd", dest="pmd", type=bool, default=False)
    parser.add_argument("--beta", dest="beta", type=float, default=0.5)
    parser.add_argument("--sampling", dest="sampling", type=bool, default=False)
    parser.add_argument("--sampling_size", dest="sampling_size", type=int, default=1)
    parser.add_argument("--constrains", dest="constrains", type=bool, default=False)
    parser.add_argument("--loaded", dest="loaded", type=bool, default=False)
    parser.add_argument("--loaded_file", dest="loaded_file", type=str, default="train_model")
    parser.add_argument("--save", dest="save", type=bool, default=False)
    parser.add_argument("--save_file", dest="save_file", type=str, default="train_model")

    args = parser.parse_args()
    main(args)
