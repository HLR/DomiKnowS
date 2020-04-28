import pickle
from argparse import ArgumentParser

from regr.sensor.allennlp.base import ReaderSensor

from sprlApp import ontology_declaration, model_declaration
from utils import seed

parser = ArgumentParser(description='Test a trained model.')
parser.add_argument(
    '--data-path', '-d',
    type=str,
    help='Path to the data.')
parser.add_argument(
    '--type', '-t',
    type=str,
    help='Dataset type (train/test).')

args = parser.parse_args()

def run(data_path, dataset_type):
    from config import Config

    graph = ontology_declaration()

    lbp = model_declaration(graph, Config.Model)

    seed()
    _, reader_sensor = next(iter(lbp.get_sensors(ReaderSensor)))
    reader = reader_sensor.reader
    dataset = reader.read(data_path, metas={'dataset_type':dataset_type})
    pkl_file = '{}.pkl'.format(data_path)
    with open(pkl_file, 'wb') as fout:
        pickle.dump(dataset, fout)
    print('Successfully dump to "{}".'.format(pkl_file))


def main():
    return run(args.data_path, args.type)


if __name__ == '__main__':
    main()
