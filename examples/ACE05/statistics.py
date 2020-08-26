from pprint import pprint
from itertools import chain, combinations
from tqdm import tqdm

from regr.graph import Concept
from regr.graph.logicalConstrain import LogicalConstrain, ifL, orL, andL

from ace05.reader import Reader, DictReader
from ace05.annotation import Entity, Timex2, Value
from ace05.graph import relations_graph, events_graph, participant_argument, attribute_argument, timex2, value
import config


def stat(path, list_path, status):
    errors = {}

    traint_reader = Reader(path, list_path=list_path, type='train', status=status)
    print('Training:', len(traint_reader))
    dev_reader = Reader(path, list_path=list_path, type='dev', status=status)
    print('Develepment:', len(dev_reader))
    test_reader = Reader(path, list_path=list_path, type='test', status=status)
    print('Testing:', len(test_reader))

    reader = Reader(path, status=status)
    span_stat = {'mentions':0, 'overlap':0, 'inclusion':0, 'same': 0, 'same/different type': 0, 'same/different subtype': 0, 'exclusion': 0}
    for data_item in tqdm(reader):
        spans = data_item['spans']
        mentions = list(chain(*(map(lambda span: map(lambda mention: (span, mention), span.mentions.values()), filter(lambda span: isinstance(span, Entity), spans.values())))))
        span_stat['mentions'] += len(mentions)
        for (span1, mention1), (span2, mention2) in combinations(mentions, r=2):
            if (mention1.extent.start == mention2.extent.start and
                mention1.extent.end == mention2.extent.end):
                span_stat['same'] += 1
                if span1.type is not span2.type:
                    span_stat['same/different type'] += 1
                if span1.subtype is not span2.subtype:
                    span_stat['same/different subtype'] += 1
            elif ((mention1.extent.end < mention2.extent.start)
                or (mention2.extent.end < mention1.extent.start)):
                span_stat['exclusion'] += 1
            elif ((mention1.extent.start < mention2.extent.start and 
                mention2.extent.start < mention1.extent.end and 
                mention1.extent.end < mention2.extent.end)
                or ((mention2.extent.start < mention1.extent.start and 
                mention1.extent.start < mention2.extent.end and 
                mention2.extent.end < mention1.extent.end))):
                span_stat['overlap'] += 1
            else:
                span_stat['inclusion'] += 1

        # relations = data_item['relations']
        # events = data_item['events']
    return span_stat


def main():
    print('Check for config:')
    print('- path:', config.path)
    print('- list_path:', config.list_path)
    print('- status:', config.status)
    result = stat(path=config.path, list_path=config.list_path, status=config.status)
    pprint(result)


if __name__ == '__main__':
    main()
