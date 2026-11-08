"""LIBERO input/scoring separation. No simulator or model imports at import time."""
import hashlib
import json
from pathlib import Path
import re


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sexpr(text):
    tokens = iter(re.findall(r'\(|\)|[^\s()]+', re.sub(r';[^\n]*', '', text)))
    def parse(first):
        if first != '(':
            return first
        result = []
        for token in tokens:
            if token == ')':
                return result
            result.append(parse(token))
        raise ValueError('unterminated BDDL expression')
    result = parse(next(tokens))
    if next(tokens, None) is not None:
        raise ValueError('trailing BDDL data')
    return result


def typed_entities(tokens):
    entities, pending = {}, []
    i = 0
    while i < len(tokens):
        if tokens[i] == '-':
            if not pending or i + 1 >= len(tokens):
                raise ValueError('malformed typed entity list')
            for name in pending:
                if name in entities:
                    raise ValueError('duplicate entity')
                entities[name] = tokens[i + 1]
            pending = []
            i += 2
        else:
            pending.append(tokens[i]); i += 1
    if pending:
        raise ValueError('untyped LIBERO entities')
    return entities


def specification(path):
    tree = sexpr(Path(path).read_text(encoding='utf-8'))
    groups = {group[0].lower(): group[1:] for group in tree if isinstance(group, list)}
    objects = typed_entities(groups[':objects'])
    fixtures = typed_entities(groups.get(':fixtures', []))
    names = sorted({**objects, **fixtures})  # independent of goals / objects of interest
    regions = {}
    for region in groups.get(':regions', []):
        attributes = {value[0]: value[1:] for value in region[1:] if isinstance(value, list)}
        target = attributes.get(':target', [None])[0]
        regions[f'{target}_{region[0]}'] = {'parent': target, 'kind': region[0]}
    if len(names) > 64:
        raise ValueError('external entity count exceeds frozen checkpoint vocabulary')
    goals = groups[':goal']
    if len(goals) == 1 and goals[0][0].lower() == 'and':
        goals = goals[0][1:]
    native_goals = goals
    normalized = []
    for goal in goals:
        if len(goal) != 3:
            raise ValueError('unsupported external goal predicate; no skill substitution allowed')
        predicate, subject, destination = goal
        if destination in regions:
            region = regions[destination]
            if predicate.lower() != 'in' or region['kind'] != 'contain_region':
                raise ValueError('unsupported goal region; no silent region-to-body approximation')
            destination = region['parent']
        normalized.append([predicate, subject, destination])
    goals = normalized
    if not goals or any(g[0].lower() not in ('on', 'in') or g[1] not in objects or g[2] not in names for g in goals):
        raise ValueError('unsupported external goal predicate; no skill substitution allowed')
    return {'instruction': ' '.join(groups[':language']), 'entities': names,
            'entity_types': {**objects, **fixtures}, 'movable': sorted(objects),
            'initial_relations': groups.get(':init', []), 'goals': goals,
            'native_goals': native_goals, 'region_parents': regions}


def generation_input(spec, images):
    # Explicit allowlist: never copy BDDL goal, task name, object-of-interest,
    # evaluation references, or an object list selected by the goal.
    ids = {name: f'entity_{index:03d}' for index, name in enumerate(spec['entities'])}
    value = {'instruction': spec['instruction'], 'entities': list(ids.values()),
             'entity_types': {ids[name]: kind for name, kind in spec['entity_types'].items()},
             'movable': [ids[name] for name in spec['movable']]}
    # Upstream region names include "target_object_region". Exposing those
    # strings would identify the answer even without the BDDL goal block.
    # Publish only relations to physical parent entities, for every object.
    relations = []
    for relation in spec['initial_relations']:
        if len(relation) != 3:
            continue
        predicate, subject, destination = relation
        destination = spec.get('region_parents', {}).get(destination, {}).get('parent', destination)
        if subject in spec['entities'] and destination in spec['entities']:
            relations.append([predicate, ids[subject], ids[destination]])
    value['initial_relations'] = sorted(relations)
    return {**value, 'images': [str(p) for p in images]}


def reference_plan(spec):
    """Calibration/scoring only. Callers must never pass this into generation."""
    result = []
    for _, source, destination in spec['goals']:
        result.extend([{'name': 'pick', 'params': {'target_entity_name': spec['entities'].index(source)}},
                       {'name': 'place', 'params': {'target_container_name': spec['entities'].index(destination)}}])
    return result


def abstract_score(plan, spec):
    relations = {tuple(str(x).lower() for x in r) for r in spec['initial_relations'] if isinstance(r, list)}
    held = None
    errors = []
    names = spec['entities']
    for index, op in enumerate(plan):
        if not isinstance(op, dict) or op.get('name') not in ('pick', 'place'):
            errors.append({'step': index, 'reason': 'unsupported_skill'}); break
        role = 'target_entity_name' if op['name'] == 'pick' else 'target_container_name'
        params = op.get('params', {})
        target = params.get(role)
        if set(params) != {role} or type(target) is not int or not 0 <= target < len(names):
            errors.append({'step': index, 'reason': 'invalid_pointer_or_schema'}); break
        name = names[target]
        if op['name'] == 'pick':
            if held is not None or name not in spec['movable']:
                errors.append({'step': index, 'reason': 'invalid_pick_precondition'}); break
            held = name
            relations = {r for r in relations if len(r) < 2 or r[1] != name.lower()}
        else:
            if held is None or held == name:
                errors.append({'step': index, 'reason': 'invalid_place_precondition'}); break
            category = spec['entity_types'][name].lower()
            relation = 'in' if any(t in category for t in ('basket', 'bowl', 'box', 'caddy')) else 'on'
            relations.add((relation, held.lower(), name.lower()))
            held = None
    goals = {tuple(str(x).lower() for x in g) for g in spec['goals']}
    satisfied = len(goals & relations)
    return {'valid_plan': not errors and held is None and bool(plan), 'violations': errors,
            'goal_recall': satisfied / len(goals), 'goal_success': not errors and satisfied == len(goals)}


def wilson(successes, count):
    if not count:
        return [0., 1.]
    z = 1.96; p = successes / count; den = 1 + z*z/count
    mid = (p + z*z/(2*count)) / den
    half = z * ((p*(1-p)/count + z*z/(4*count*count)) ** .5) / den
    return [max(0., mid-half), min(1., mid+half)]
