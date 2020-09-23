from ace05.graph import graph
from ace05.reader import Reader, DictReader, DictParagraphReader

from model import model
import config

from dummy import DummyReader, DummyDictReader

def main():
    program = model(graph)
    traint_reader = DictParagraphReader(config.path, list_path=config.list_path, type='train', status=config.status)
    for item in traint_reader:
        print(item)
    for node in program.populate(traint_reader, device='auto'):
        print(node)
        tokens = node.findDatanodes(select=graph['linguistic/token'])
        spans = node.findDatanodes(select=graph['linguistic/span'])
        span_annotations = node.findDatanodes(select=graph['linguistic/span_annotation'])
        print(len(tokens), len(spans), len(span_annotations))

if __name__ == "__main__":
    main()
