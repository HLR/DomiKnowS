from ace05.graph import graph
from ace05.reader import Reader, DictReader

from model_prototype import model
import config


def main():
    program = model(graph)
    #traint_reader = Reader(config.path, list_path=config.list_path, type='train', status=config.status)
    traint_reader = [{}]  # dummy data
    for node in program.populate(traint_reader, device='auto'):
        print(node)

if __name__ == "__main__":
    main()
