import os
import pickle
from argparse import ArgumentParser

from sprlApp import ontology_declaration, model_declaration
from utils import seed

parser = ArgumentParser(description='Test a trained model.')
parser.add_argument('--model-path', '-m', type=str,
                    help='Path to the model serialization_dir.')
parser.add_argument('--model', '-M', type=str, default='last',
                    help='Model id, last, or best.')
parser.add_argument('--data-path', '-d', type=str,
                    help='Path to the data.')
parser.add_argument('--log-solver', '-s', type=str, default='solver.log',
                    help='Path save inference log.')
parser.add_argument('--batch-size', '-b', type=int, default=1,
                    help='Batch size in testing.')

args = parser.parse_args()

def test(model_path, model, data_path, log_solver, batch_size):
    #with open(os.path.join(model_path, 'config.pkl'), 'rb') as fin:
    #    Config = pickle.load(fin)
    from config import Config

    graph = ontology_declaration()

    Config.Model.log_dir = '.'
    lbp = model_declaration(graph, Config.Model)

    seed()
    lbp.load(model_path, model)
    metrics = lbp.test(data_path, log_solver, batch_size)
    print(metrics)


def main():
    return test(args.model_path, args.model, args.data_path, args.log_solver, args.batch_size)


if __name__ == '__main__':
    main()
