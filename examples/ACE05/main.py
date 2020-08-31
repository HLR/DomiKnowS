from ace05.graph import graph
from ace05.reader import Reader, DictReader

from model import model
import config


def main():
    program = model(graph)
    #traint_reader = Reader(config.path, list_path=config.list_path, type='train', status=config.status)
    class DummySpan:
        class Mention:
            def __init__(self, start, end):
                self.start = start
                self.end = end
        def __init__(self, start, end):
            self.mentions = []
            self.mentions.append(self.Mention(start, end))
    traint_reader = [{
        'text': 'John works for IBM.',
        'spans': [DummySpan(0,1), DummySpan(3,4)]}]  # dummy data
    for node in program.populate(traint_reader, device='auto'):
        print(node)

if __name__ == "__main__":
    main()
