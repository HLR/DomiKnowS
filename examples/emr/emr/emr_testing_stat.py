import os
import pickle
import logging
from argparse import ArgumentParser
import importlib
import torch
import numpy as np


if __package__ is None or __package__ == '':
    from emr_full import ontology_declaration, model_declaration
    from utils import seed, Namespace
else:
    from .emr_full import ontology_declaration, model_declaration
    from .utils import seed, Namespace


parser = ArgumentParser(description='Test a trained model.')
parser.add_argument('--model-path', '-m', type=str,
                    help='Path to the model serialization_dir.')
parser.add_argument('--config-path', '-c', type=str,
                    help='override default config path to the config.pkl in model serialization_dir.')
parser.add_argument('--config-path-py', '-C', type=str,
                    help='override default config to load from a Config instance python module than config.pkl.')
parser.add_argument('--vocab-path', '-v', type=str,
                    help='override default vocabulary path than vocab in model serialization_dir.')
parser.add_argument('--data-path', '-d', type=str,
                    help='Path to the data.')
parser.add_argument('--solver-log', '-s', type=str, default='solver.log',
                    help='Path save inference log.')
parser.add_argument('--batch-size', '-b', type=int, default=1,
                    help='Batch size in testing.')

args = parser.parse_args()


def test(model_path, vocab_path, data_path, solver_log, batch_size, Config):
    graph = ontology_declaration()

    viol_logger = logging.getLogger('regr.graph.allennlp.utils.violation')
    viol_logger.propagate = False
    fh = logging.FileHandler('violations.log', 'w')
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    viol_logger.addHandler(fh)
    viol_logger.setLevel(logging.DEBUG)

    Config = Namespace(Config)
    Config.Model.graph.inference_testing_set = False
    Config.Model.graph.inference_loss = False
    Config.Model.graph.inference_interval = 100

    lbp = model_declaration(graph, Config.Model)

    seed()
    lbp.load(model_path, 'last', vocab_path)
    metrics = lbp.test(data_path, solver_log, batch_size, log_violation=True)
    print(metrics)


def main():
    if args.config_path_py:
        config = importlib.import_module(args.config_path_py)
        Config = config.Config
    else:
        if args.config_path:
            config_path = args.config_path
        else:
            config_path = os.path.join(args.model_path, 'config.pkl')
        with open(config_path, 'rb') as fin:
            Config = pickle.load(fin)
    return test(args.model_path, args.vocab_path, args.data_path, args.solver_log, args.batch_size, Config)


if __name__ == '__main__':
    main()
