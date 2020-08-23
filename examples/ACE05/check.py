from pprint import pprint
from itertools import chain
from tqdm import tqdm

from regr.graph.logicalConstrain import ifL, orL, andL

from ace05.reader import Reader, DictReader
from ace05.annotation import Entity, Timex2, Value
from ace05.graph import relations_graph, events_graph, participant_argument, attribute_argument, timex2, value
import config


def compile_event_rules(events_graph):
    event_rules = {}
    for _, constraint in events_graph.logicalConstrains.items():
        if not isinstance(constraint, ifL):
            continue
        role_argument, xy, implication = constraint.e
        rel = role_argument.is_a()[0]
        assert rel.src == role_argument
        argument_type = rel.dst
        x, y = xy
        assert isinstance(implication, andL)
        event, x_, *argument_implication = implication.e
        if len(argument_implication) == 1:  # orL
            assert isinstance(argument_implication[0], orL)
            *argument_types, y_ = argument_implication[0].e
        elif len(argument_implication) == 2:  # type, y
            argument_type, y_ = argument_implication
            argument_types = [argument_type]
        else:
            assert False
        assert x == x_ and y == y_
        role = role_argument.name[len(event.name)+1:]
        event_rules[event, role] = argument_types
    return event_rules


def check(event_rules, path, list_path, status, known_errors={}, collect_errror=True):
    errors = {}
    if collect_errror:
        def check_event_arg(event, arg):
            if arg.ref.type not in entity_types:
                errors.setdefault(f'{event.subtype.name}: {arg.role} is {arg.ref.type}', []).append((event_id, arg.refid))
    else:
        def check_event_arg(event, arg):
            assert arg.ref.type not in entity_types

    traint_reader = Reader(path, list_path=list_path, type='train', status=status)
    print('Training:', len(traint_reader))
    dev_reader = Reader(path, list_path=list_path, type='dev', status=status)
    print('Develepment:', len(dev_reader))
    test_reader = Reader(path, list_path=list_path, type='test', status=status)
    print('Testing:', len(test_reader))
    reader = Reader(path, status=status)
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
                if (event_id, arg.refid) in known_errors['event-arg']: continue
                try:
                    role = arg.role
                    if role.startswith('Time-'):
                        role = 'Time'
                    entity_types = event_rules[event.subtype, role]
                    assert isinstance(arg.ref, (Entity, Timex2, Value))
                    check_event_arg(event, arg)
                except:
                    errors.setdefault(f'{event.subtype.name}: {arg.role} is {arg.ref.type}', []).append((event_id, arg.refid))
                    if not collect_errror:
                        raise
            # if there is event.subtype, then event.subtype is a event.type
            assert not event.subtype or event.type in set(map(lambda e: e.dst, event.subtype.is_a()))
    print(f'Checked {len(reader)} examples.')
    print(f'Collected {len(list(chain(*errors.values())))} new errors.')
    return errors


def main():
    event_rules = compile_event_rules(events_graph)
    print('Check for config:')
    print('- path:', config.path)
    print('- list_path:', config.list_path)
    print('- status:', config.status)
    print('- known errors:', len(list(chain(*config.known_errors.values()))))
    errors = check(event_rules, path=config.path, list_path=config.list_path, status=config.status, known_errors=config.known_errors)
    pprint(errors)


if __name__ == '__main__':
    main()
