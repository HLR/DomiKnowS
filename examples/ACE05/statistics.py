from pprint import pprint
from itertools import chain, combinations
from tqdm import tqdm
import numpy as np

from regr.graph import Concept
from regr.graph.logicalConstrain import LogicalConstrain, ifL, orL, andL

from ace05.reader import Reader, DictReader
from ace05.reader import ParagraphReader as Reader, DictParagraphReader as DictReader
from ace05.annotation import Entity, Timex2, Value
from ace05.graph import relations_graph, events_graph, participant_argument, attribute_argument, timex2, value
import config

def get_all_mentions(data_item):
    spans = data_item['spans']
    mentions = list(chain(*(map(lambda span: map(lambda mention: (span, mention), span.mentions.values()), spans.values()))))
    return mentions

def get_entity_mentions(data_item):
    spans = data_item['spans']
    mentions = list(chain(*(map(lambda span: map(lambda mention: (span, mention), span.mentions.values()), filter(lambda span: isinstance(span, Entity), spans.values())))))
    return mentions

def get_mentions_in_event(data_item):
    spans = data_item['spans']
    events = data_item['events']
    in_event = set()
    for emention in chain(*(event.mentions.values() for event in events.values())):
        for argument in emention.arguments:
            in_event.add(argument.ref)
    def is_entity(span):
        return isinstance(span, Entity)
    spans = filter(is_entity, spans.values())
    mentions = list(chain(*(map(lambda span: map(lambda mention: (span, mention), span.mentions.values()), spans))))
    def is_in_event(span_mention):
        span, mention = span_mention
        return mention in in_event
    mentions = list(filter(is_in_event, mentions))
    return mentions

get_mentions = get_all_mentions # get_all_mentions, get_entity_mentions, get_mentions_in_event

def stat(path, list_path, status):
    errors = {}

    traint_reader = Reader(path, list_path=list_path, type='train', status=status)
    print('Training:', len(traint_reader))
    dev_reader = Reader(path, list_path=list_path, type='dev', status=status)
    print('Develepment:', len(dev_reader))
    test_reader = Reader(path, list_path=list_path, type='test', status=status)
    print('Testing:', len(test_reader))

    reader = Reader(path, status=status)
    span_stat = {'mentions':0, 'mentions/overlap':0, 'mentions/inclusion':0, 'mentions/exclusion': 0, 'mentions/same': 0, 'same/different basetype': 0, 'same/different type': 0, 'same/different subtype': 0, 'length/max': 0, 'length/max/example': None}
    length_list = []
    head_length_list = []
    for data_item in tqdm(reader):
        mentions = get_mentions(data_item)
        span_stat['mentions'] += len(mentions)
        for _, mention in mentions:
            length = mention.extent.end - mention.extent.start
            length_list.append(length)
            head_length = mention.head.end - mention.head.start
            head_length_list.append(head_length)
            if length > span_stat['length/max']:
                span_stat['length/max'] = length
                span_stat['length/max/example'] = data_item['text'] + '-'*40 + '\n' + mention.extent.text + '-'*40 + '\n' + mention.head.text
        for span1, mention1 in mentions:
            same = False
            exclusion = False
            overlap = False
            inclusion = False
            for span2, mention2 in mentions:
                if span2 is span1:
                    continue
                if (mention1.extent.start == mention2.extent.start and
                    mention1.extent.end == mention2.extent.end):
                    same = True
                    if span1.basetype != span2.basetype:
                        span_stat['same/different basetype'] += 0.5
                    if span1.type is not span2.type:
                        span_stat['same/different type'] += 0.5
                    if span1.subtype is not span2.subtype:
                        span_stat['same/different subtype'] += 0.5
                elif ((mention1.extent.end < mention2.extent.start)
                    or (mention2.extent.end < mention1.extent.start)):
                    exclusion = True
                elif ((mention1.extent.start < mention2.extent.start and 
                    mention2.extent.start < mention1.extent.end and 
                    mention1.extent.end < mention2.extent.end)
                    or ((mention2.extent.start < mention1.extent.start and 
                    mention1.extent.start < mention2.extent.end and 
                    mention2.extent.end < mention1.extent.end))):
                    overlap = True
                else:
                    inclusion = True
            span_stat['mentions/same'] += same
            span_stat['mentions/overlap'] += overlap
            span_stat['mentions/inclusion'] += inclusion
            span_stat['mentions/exclusion'] += not(same or overlap or inclusion)

    span_stat['length/extend'] = np.histogram(length_list)
    span_stat['length/head'] = np.histogram(head_length_list)
    del span_stat['length/max/example']
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
