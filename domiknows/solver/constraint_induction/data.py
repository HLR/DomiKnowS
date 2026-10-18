"""Annotation-aware observations and three-valued candidate validation."""
import json
import math
from numbers import Integral

import torch
from domiknows.graph import EnumConcept
from domiknows.solver.explanation.evidence import native, graph_path
from domiknows.solver.explanation.session import _clone
from domiknows.solver.incremental.session import _materialize
from .specs import root_concept


def sample_nodes(root):
    seen, result, pending = set(), [], [root]
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node)); result.append(node)
        pending.extend(target for targets in node.relationLinks.values() for target in targets
                       if hasattr(target, 'ontologyNode'))
    return result


def sample_key(value):
    if isinstance(value, bool) or not isinstance(value, (str, Integral)):
        raise ValueError('Sample IDs must be strings or integers')
    return json.dumps(native(value))


def exact_attribute(node, concept, suffix):
    """Do not use getAttribute's fuzzy suffix fallback for missing labels."""
    names = (f'<{concept.name}>/{suffix}', f'{concept.name}/{suffix}')
    for name in names:
        if name in node.attributes:
            return node.attributes[name]
    root = node.attributes.get('rootDataNode')
    if root is not None:
        for container_name in ('variableSet', 'propertySet'):
            container = root.attributes.get(container_name, {})
            for name in names:
                key = node.ontologyNode.name + '/' + name
                if key in container and type(node.instanceID) is int:
                    return container[key][node.instanceID]
    return None


def label_value(raw, concept, prediction=False):
    if raw is None:
        return None
    if prediction:
        values = torch.as_tensor(raw).detach().reshape(-1)
        expected = len(concept.enum) if isinstance(concept, EnumConcept) else 2
        if values.numel() != expected or not bool(torch.isfinite(values).all()):
            return None
        index = int(values.argmax())
        return concept.enum[index] if isinstance(concept, EnumConcept) else bool(index)
    if hasattr(raw, 'tobytes') and hasattr(raw, 'tolist'):
        raw = raw.tolist()
    value = native(raw.detach().reshape(-1) if torch.is_tensor(raw) else raw)
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    if value is None or isinstance(value, list) and any(v is None for v in value):
        return None
    if isinstance(value, list):
        if len(value) == 1:
            value = value[0]
        else:
            expected = len(concept.enum) if isinstance(concept, EnumConcept) else 2
            if len(value) != expected or any(v not in (0, 1, False, True) for v in value) or sum(value) != 1:
                raise ValueError('Annotations must be Boolean, class indices/names, or one-hot vectors')
            value = value.index(1)
    if isinstance(concept, EnumConcept):
        if isinstance(value, str) and value in concept.enum:
            return value
        if type(value) is int and 0 <= value < len(concept.enum):
            return concept.enum[value]
    elif value in (False, True, 0, 1):
        return bool(value)
    raise ValueError('Invalid annotation value')


def extract(examples, vocabulary, identify, complete, key):
    result = []
    for source in examples:
        sample = identify(source)
        sample_key(sample)
        root = _materialize(_clone(source))
        nodes = sample_nodes(root)
        domains, values, links = {}, {}, {}
        for path, concept in vocabulary.concepts.items():
            domain = root_concept(concept)
            matched = [n for n in nodes if n.ontologyNode is domain]
            domains[path] = []
            for node in matched:
                entity = json.dumps({'path': graph_path(domain), 'instance_id': native(node.instanceID)}, sort_keys=True)
                if entity in domains[path]:
                    raise ValueError('Duplicate grounded entity identifiers')
                domains[path].append(entity)
                values[(path, entity)] = dict(annotation=label_value(exact_attribute(node, concept, 'label'), concept),
                    prediction=label_value(exact_attribute(node, concept, '/'.join(key)), concept, prediction=True))
                links[entity] = {name: [json.dumps({'path': graph_path(n.ontologyNode),
                    'instance_id': native(n.instanceID)}, sort_keys=True) for n in targets]
                    for name, targets in node.relationLinks.items()}
        # Include owner domains even when their root concepts have no predictions.
        for node in nodes:
            path = graph_path(node.ontologyNode)
            entity = json.dumps({'path': path, 'instance_id': native(node.instanceID)}, sort_keys=True)
            domains.setdefault(path, [])
            if entity not in domains[path]:
                domains[path].append(entity)
        result.append(dict(sample_id=native(sample), domains=domains, values=values, links=links,
                           complete=set(complete.get(sample, ()))))
    return result


def literal_value(sample, literal, entity, source):
    value = sample['values'].get((literal['concept'], entity), {}).get(source)
    if value is None:
        return None
    return value == literal['class'] if literal['class'] is not None else bool(value)


def evaluate(sample, spec, vocabulary, source):
    """Yield observations: applicable True/False/None, violation, grounded data."""
    kind = spec['template']
    if kind in ('implication', 'disjointness'):
        a, b = (spec['antecedent'], spec['consequent']) if kind == 'implication' else (spec['left'], spec['right'])
        for entity in sample['domains'][a['concept']]:
            av, bv = literal_value(sample, a, entity, source), literal_value(sample, b, entity, source)
            if kind == 'implication':
                applicable = False if av is False else None if av is None or bv is None else True
                violated = av is True and bv is False
            else:
                applicable = None if av is None or bv is None else av or bv
                violated = av is True and bv is True
            yield applicable, violated, dict(entity=json.loads(entity), values=[av, bv])
    elif kind == 'relation_typing':
        relation = spec['relation']
        for entity in sample['domains'][relation['concept']]:
            av = literal_value(sample, relation, entity, source)
            role_values, bound = {}, {}
            for role, literal in spec['roles'].items():
                targets = sample['links'].get(entity, {}).get(role, ())
                bound[role] = [json.loads(t) for t in targets]
                role_values[role] = literal_value(sample, literal, targets[0], source) if len(targets) == 1 else None
            applicable = False if av is False else None if av is None or any(v is None for v in role_values.values()) else True
            yield applicable, av is True and any(v is False for v in role_values.values()), dict(
                entity=json.loads(entity), roles=bound, values={'relation': av, **role_values})
    else:
        literal = spec['predicate']
        entities = sample['domains'][literal['concept']]
        if literal['concept'] not in sample['complete']:
            yield None, False, {'domain': 'completeness not declared'}
            return
        groups = {'example': entities}
        if spec['group_by'] is not None:
            roles = root_concept(vocabulary.concepts[literal['concept']])._out['has_a']
            role = next(r for r in roles if r.name == spec['group_by'])
            groups = {owner: [] for owner in sample['domains'].get(graph_path(root_concept(role.dst)), ())}
            for entity in entities:
                owners = sample['links'].get(entity, {}).get(role.name, ())
                if len(owners) != 1 or owners[0] not in groups:
                    yield None, False, {'domain': 'missing or ambiguous grouping links'}
                    return
                groups[owners[0]].append(entity)
        for owner, group in groups.items():
            values = [literal_value(sample, literal, entity, source) for entity in group]
            if any(v is None for v in values):
                yield None, False, dict(group=owner, count=None)
            else:
                count = sum(values)
                violated = count > spec['limit'] if spec['operator'] == 'at_most' else count < spec['limit']
                yield True, violated, dict(group=owner, count=count, limit=spec['limit'],
                                          entities=[json.loads(e) for e in group])
