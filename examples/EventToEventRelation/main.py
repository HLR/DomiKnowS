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
from doc_reader import load_dataset


def main(args):
    from graph import graph, paragraph, paragraph_contain, event_relation, relation_classes
    # Set the cuda number we want to use
    cuda_number = args.cuda_number
    cur_device = "cuda:" + str(cuda_number) if torch.cuda.is_available() else 'cpu'

    train_dataset, valid_dataset, test_dataset = load_dataset(batch_size=2)

    # Declare Program
    program = program_declaration(cur_device)

    program.train(train_dataset, valid_set=valid_dataset, test_set=test_dataset, train_epoch_num=args.epoch,
                  Optim=lambda params: torch.optim.Adam(params, lr=args.learning_rate, amsgrad=True)
                  ,device=cur_device)

    # TODO: Evaluate Testing dataset
    print(program.model.loss)
    for datanode in program.populate(test_dataset, device=cur_device):
        for event_relation in datanode.getChildDataNodes():
            print(event_relation)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Event to Event Relation Learning Code")

    parser.add_argument('--cuda', dest='cuda_number', default=0, help='cuda number to train the models on', type=int)
    parser.add_argument('--epoch', dest='epoch', default=5, help='number of epoch to train the model', type=int)
    parser.add_argument('--lr', dest='learning_rate', default=1e-3, help='learning rate of the model', type=float)

    args = parser.parse_args()
    main(args)

