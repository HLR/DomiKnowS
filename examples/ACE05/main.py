from tqdm import tqdm

from ace05.reader import Reader, DictReader
from ace05.annotation import Entity, Timex2, Value
from ace05.graph import timex2, value
from ace05.errors import KNOWN_ERRORS_TIMEX2NORM as KNOWN_ERRORS
import config


# errors = {}


def main():
    traint_reader = Reader(config.path, list_path='data_list.csv', type='train', status='timex2norm')
    print('Training:', len(traint_reader))
    dev_reader = Reader(config.path, list_path='data_list.csv', type='dev', status='timex2norm')
    print('Develepment:', len(dev_reader))
    test_reader = Reader(config.path, list_path='data_list.csv', type='test', status='timex2norm')
    print('Testing:', len(test_reader))
    reader = Reader(config.path, status='timex2norm')
    for data_item in tqdm(reader):
        text = data_item['text']
        spans = data_item['referables']
        relations = data_item['relations']
        events = data_item['events']
        # validate relations
        for rel_id, rel in relations.items():
            # relation has two arguments
            assert rel.arguments[0] is not None and rel.arguments[1] is not None
            # relation type match entity type
            # TODO: there are constraints arg[1] can be one of A, B, or C, which cannot be checked with graph
            # if there is rel.subtype, then rel.subtype is a rel.type
            assert not rel.subtype or rel.type in set(map(lambda r: r.dst, rel.subtype.is_a()))
            # 
        for event_id, event in events.items():
            # event arguments
            for arg in event.arguments:
                if (event_id, arg.refid) in KNOWN_ERRORS['event-arg']: continue
                if isinstance(arg.ref, Entity):
                    assert arg.ref.type in set(map(lambda e: e.dst, event.subtype.involve())), f'Value type mismatch in {(event_id, arg.refid)}: {event.subtype.name}: {arg.role} is {arg.ref.type}'
                    # if arg.ref.type not in set(map(lambda e: e.dst, event.subtype.involve())):
                    #     errors.setdefault(f'{event.subtype.name}: {arg.role} is {arg.ref.type}', []).append((event_id, arg.refid))
                elif isinstance(arg.ref, Timex2):
                    assert timex2 in set(map(lambda e: e.dst, event.subtype.involve()))
                elif isinstance(arg.ref, Value):
                    assert arg.ref.type in set(map(lambda e: e.dst.name, event.subtype.involve())).intersection(set(map(lambda e: e.src.name, value._in['is_a']))), f'Value type mismatch in {(event_id, arg.refid)}: {event.subtype.name}: {arg.role} is {arg.ref.type}'
                    # if arg.ref.type not in set(map(lambda e: e.dst.name, event.subtype.involve())).intersection(set(map(lambda e: e.src.name, value._in['is_a']))):
                    #     errors.setdefault(f'{event.subtype.name}: {arg.role} is {arg.ref.type}', []).append((event_id, arg.refid))
                else:
                    assert False, f'Unsupported argument type {type(arg.ref)}'
            # if there is event.subtype, then event.subtype is a event.type
            assert not event.subtype or event.type in set(map(lambda e: e.dst, event.subtype.is_a()))
            #
    print(f'Checked {len(reader)} examples.')


if __name__ == '__main__':
    main()
    # print(errors)
