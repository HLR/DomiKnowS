import torch
import numpy as np
import argparse
from dataset import load_annodata
from model import model_declaration
from graph import Category
from graph import graph
import logging


def main():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = 'cpu'
    print("selected device is:", device)

    data = load_annodata("data/LowContextAnnoData.csv")

    train_data, val_data, test_data = np.split(data, [int(len(data) * 0.1), int(len(data) * 0.9)])

    program = model_declaration(device)

    logging.basicConfig(level=logging.INFO)

    program.train(train_data, valid_set=val_data, train_epoch_num=4,
                  Optim=lambda param: torch.optim.Adam(param, lr=5e-6), device=device)
    program.test(test_data)
    # program.populate(test_data, device=device)
    # for node in program.populate(test_data, device=device):
    #     print(node)
    #     print(node.findDatanodes(select=graph['TextSequence']))
    #     print(node.findDatanodes(select=graph['ParentTag']))
    #     print(node.findDatanodes(select=graph['Category']))
    # print(node.findDatanodes(select=graph[Category]))
    # print(node.findDatanodes(select=graph['ParentTag']))
    # tokens = node.findDatanodes(select=graph['linguistic/token'])
    # spans = node.findDatanodes(select=graph['linguistic/span'])
    # span_annotations = node.findDatanodes(select=graph['linguistic/span_annotation'])
    # print(len(tokens), len(spans), len(span_annotations))


if __name__ == '__main__':
    main()
