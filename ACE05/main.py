import logging
import torch

from ace05.graph import graph
from ace05.reader import Reader, DictReader, DictParagraphReader

from model import model
import config


logging.basicConfig(level=logging.INFO)


def main():
    program = model(graph)
    train_reader = DictParagraphReader(config.path, list_path=config.list_path, type='train', status=config.status)
    # for item in traint_reader:
    #     print(item)
    program.train(train_reader, train_epoch_num=2, Optim=lambda param: torch.optim.SGD(param, lr=.001))
    program.test(train_reader)
    for node in program.populate(train_reader, device='auto'):
        print(node)
        tokens = node.findDatanodes(select=graph['linguistic/token'])
        spans = node.findDatanodes(select=graph['linguistic/span'])
        span_annotations = node.findDatanodes(select=graph['linguistic/span_annotation'])
        print(len(tokens), len(spans), len(span_annotations))


if __name__ == "__main__":
    main()
