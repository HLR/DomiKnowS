from pprint import pprint
from itertools import chain
from tqdm import tqdm

from regr.graph import Concept
from regr.graph.logicalConstrain import LogicalConstrain, ifL, orL, andL

from ace05.reader import Reader, DictReader
from ace05.reader import ParagraphReader as Reader, DictParagraphReader as DictReader
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
        assert xy == x_ + y_
        role = role_argument.name[len(event.name)+1:]
        event_rules[event, role] = argument_types
    return event_rules


def compile_relation_rules(relations_graph):
    relation_rules = {}
    for _, constraint in relations_graph.logicalConstrains.items():
        if not isinstance(constraint, ifL):
            continue
        relation, xy, implication = constraint.e
        relation_rules[relation] = (relation, xy, implication)
    return relation_rules


def validate_rel_arg(rel, *args, relation, xy, implication):
    if isinstance(implication, LogicalConstrain):
        impls = iter(implication.e)
        new_impls = []
        for impl in impls:
            if isinstance(impl, LogicalConstrain):
                new_impls.append(impl)
                continue
            impl_group = []
            while not isinstance(impl, tuple):
                impl_group.append(impl)
                impl = next(impls)
            impl_group.append(impl)
            new_impls.append(impl_group)
        vals = [validate_rel_arg(rel, *args, relation=relation, xy=xy, implication=impl) for impl in new_impls]
        if isinstance(implication, andL):
            return all(vals)
        elif isinstance(implication, orL):
            return any(vals)
        else:
            raise ValueError(f'Implication of relation {relation} not recognized: {implication}')
    elif isinstance(implication, (list, tuple)):
        *concepts, arg_tuple = implication
        assert all(isinstance(concept, Concept) for concept in concepts)
        assert len(arg_tuple) == 1
        arg_str = arg_tuple[0]
        assert arg_str in xy
        index = xy.index(arg_str)
        arg = args[index]
        assert arg.role == f'Arg-{index+1}'
        assert isinstance(arg.ref, Entity)
        return arg.ref.type in concepts or arg.ref.subtype in concepts  # Citizen-Resident-Religion-Ethnicity Arg-2: PER.Group
    else:
        raise ValueError(f'Implication of relation {relation} not recognized: {implication}')


def check(relation_rules, event_rules, path, list_path, status, known_errors={}, stop_on_errror=False):
    errors = {}

    traint_reader = Reader(path, list_path=list_path, type='train', status=status)
    print('Training:', len(traint_reader))
    dev_reader = Reader(path, list_path=list_path, type='dev', status=status)
    print('Develepment:', len(dev_reader))
    test_reader = Reader(path, list_path=list_path, type='test', status=status)
    print('Testing:', len(test_reader))
    reader = Reader(path, status=status)
    for data_item in tqdm(reader):
        text = data_item['text']
        spans = data_item['spans']
        relations = data_item['relations']
        events = data_item['events']
        # validate relations
        for rel_id, rel in relations.items():
            # relation has two arguments
            assert rel.arguments[0] is not None and rel.arguments[1] is not None
            # relation type match entity type
            if rel.subtype is not None:
                relation, xy, implication = relation_rules[rel.subtype]
                try:
                    assert validate_rel_arg(rel, *rel.arguments, relation=relation, xy=xy, implication=implication), f'{rel.subtype.name}: Arg-1 is {rel.arguments[0].ref.type}, Arg-2 is {rel.arguments[1].ref.type}'
                except Exception as e:
                    message = str(e)
                    if message in known_errors['relation-arg'] and rel_id in known_errors['relation-arg'][message]:
                        continue
                    errors.setdefault(message, []).append(rel_id)
                    if stop_on_errror:
                        raise
            # if there is rel.subtype, then rel.subtype is a rel.type
            assert not rel.subtype or rel.type in set(map(lambda r: r.dst, rel.subtype.is_a()))
            # 
        for event_id, event in events.items():
            # event arguments
            for arg in event.arguments:
                try:
                    role = arg.role
                    if role.startswith('Time-'):
                        role = 'Time'
                    try:
                        entity_types = event_rules[event.subtype, role]
                    except KeyError as e:
                        key = e.args[0]
                        concept, role = key
                        raise KeyError(f'{role} is not a valid role in {concept.name}.')
                    assert isinstance(arg.ref, (Entity, Timex2, Value))
                    assert arg.ref.type in entity_types, f'{event.subtype.name}: {arg.role} is {arg.ref.type}'
                except Exception as e:
                    message = str(e)
                    if message in known_errors['event-arg'] and (event_id, arg.refid) in known_errors['event-arg'][message]:
                        continue
                    errors.setdefault(message, []).append((event_id, arg.refid))
                    if stop_on_errror:
                        raise
            # if there is event.subtype, then event.subtype is a event.type
            assert not event.subtype or event.type in set(map(lambda e: e.dst, event.subtype.is_a()))
    print(f'Checked {len(reader)} examples.')
    print(f'Collected {len(list(chain(*errors.values())))} new errors.')
    return errors


def main():
    relation_rules = compile_relation_rules(relations_graph)
    event_rules = compile_event_rules(events_graph)
    print('Check for config:')
    print('- path:', config.path)
    print('- list_path:', config.list_path)
    print('- status:', config.status)
    print('- known errors:')
    for error_type, errors in config.known_errors.items():
        print(f'  - {error_type}:', len(list(chain(*errors.values()))))
    errors = check(relation_rules, event_rules, path=config.path, list_path=config.list_path, status=config.status, known_errors=config.known_errors, stop_on_errror=False)
    pprint(errors)


if __name__ == '__main__':
    main()
